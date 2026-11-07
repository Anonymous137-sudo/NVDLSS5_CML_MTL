"""ANE export adapter for the full recovered graph; no weight/block reduction.

Run --verify first. The vendor export uses a tracing-only standard softmax;
this adapter instead expresses its half-bit affine approximation arithmetically.
Window partition/reverse use at most rank five, following Apple's vision ANE
guidance. Primitive parity does not establish full-network Core ML parity.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'MLX-DLSS/python'))
import torch
from mlxdlss import model as reference
original_round_trip = reference.e4m3_round_trip
COMPACT_ARITHMETIC = False


def scalar_round_trip(value):
    if not torch.jit.is_tracing():
        return original_round_trip(value)
    magnitude = value.abs().clamp(max=448.0)
    exponent = torch.floor(torch.log2(magnitude.clamp(min=2**-6)))
    step = torch.where(magnitude < 2**-6, 2**-9, torch.pow(2.0, exponent - 3))
    rounded = torch.round(magnitude / step) * step
    return torch.where(value < 0, -rounded, rounded)


def cosine_normalize(value):
    floor = torch.tensor(reference.COSINE_NORM_FLOOR, dtype=torch.float16, device=value.device)
    if value.shape[-1] != 32:
        h = value.half()
        norm = h.square().sum(dim=-1, keepdim=True, dtype=torch.float16)
        return (h * torch.maximum(norm, floor).rsqrt()).to(value.dtype)
    h = value.half()
    a, b, c, d = h[..., :8], h[..., 8:16], h[..., 16:24], h[..., 24:]
    first = reference._half_fma(b, b, reference._half_multiply(a, a))
    second = reference._half_fma(d, d, reference._half_multiply(c, c))
    partial = reference._half_add(first, second)
    even = reference._half_add(partial[..., :2], partial[..., 4:6])
    odd = reference._half_add(partial[..., 2:4], partial[..., 6:8])
    pair = reference._half_add(even, odd)
    norm = torch.maximum(reference._half_add(pair[..., 0], pair[..., 1]), floor)
    reciprocal = norm.float().rsqrt().half().unsqueeze(-1)
    return reference._half_multiply(h, reciprocal).to(value.dtype)


def install_adapters(compact):
    global COMPACT_ARITHMETIC
    COMPACT_ARITHMETIC = compact
    reference.partition_windows = partition
    reference.reverse_windows = reverse
    reference.vendor_approximate_softmax = affine_softmax
    if compact:
        reference.e4m3_round_trip = scalar_round_trip
        reference.vendor_cosine_normalize = cosine_normalize


def partition(value, window_size):
    n, h, w, c = value.shape
    s = window_size
    return (value.reshape(n, h // s, s, w, c).permute(0, 1, 3, 2, 4)
            .reshape(n * (h // s), w // s, s, s, c)
            .permute(0, 1, 3, 2, 4).reshape(-1, s * s, c))


def reverse(windows, *, batch_count, height, width, window_size):
    n, h, w, s, c = batch_count, height, width, window_size, windows.shape[-1]
    return (windows.reshape(n * (h // s), w // s, s, s, c)
            .permute(0, 1, 3, 2, 4).reshape(n, h // s, w, s, c)
            .permute(0, 1, 3, 2, 4).reshape(n, h, w, c))


def affine_softmax(value):
    affine = (value.to(torch.float16).float() * 0.044921875 + 1.30078125).half()
    affine = affine.clamp(1.03125, 1.5693359375).float()
    # Here affine is an exactly representable half in [1,2); its half bits
    # are 15360 + 1024*(affine-1). The vendor packed-pair transform has a
    # constant carry of eight over this clamped range, so each resulting
    # half bit pattern is independently 32*(bits-15360).
    bits = (affine - 1) * 32768
    exponent = torch.floor(bits / 1024)
    mantissa = bits - exponent * 1024
    base = 2.0 if COMPACT_ARITHMETIC else torch.full_like(bits, 2)
    normal = (1 + mantissa / 1024) * torch.pow(base, exponent - 15)
    weights = torch.where(bits < 1024, bits * (2**-24), normal).half()
    totals = weights.sum(dim=-1, keepdim=True, dtype=torch.float16)
    reciprocal = totals.float().reciprocal().half()
    return reference.e4m3_round_trip((weights * reciprocal).to(value.dtype))


def verify():
    torch.manual_seed(7)
    cases = []
    for shape, size in [((1, 32, 48, 32), 8), ((2, 16, 32, 64), 16)]:
        x = torch.arange(torch.tensor(shape).prod()).reshape(shape)
        expected = reference.partition_windows(x, size)
        actual = partition(x, size)
        assert torch.equal(actual, expected)
        assert torch.equal(reverse(actual, batch_count=shape[0], height=shape[1], width=shape[2], window_size=size), x)
        cases.append({'shape': shape, 'window': size, 'partition_exact': True, 'reverse_exact': True})
    maximum = 0.0
    for dtype in [torch.float16, torch.float32]:
        for width in [64, 256, 960]:
            x = (torch.randn(2, 3, width, width) * 12).to(dtype)
            expected, actual = reference.vendor_approximate_softmax(x), affine_softmax(x)
            maximum = max(maximum, (actual - expected).abs().max().item())
            assert torch.equal(actual, expected), (dtype, width, maximum)
        q = torch.randn(3, 4, 64, 32).to(dtype)
        assert torch.equal(cosine_normalize(q), reference.vendor_cosine_normalize(q))
        x = (torch.randn(256, 256) * 100).to(dtype)
        traced_round = torch.jit.trace(scalar_round_trip, x)
        assert torch.equal(traced_round(x), original_round_trip(x))
        traced_norm = torch.jit.trace(cosine_normalize, q)
        assert torch.equal(traced_norm(q), reference.vendor_cosine_normalize(q))
    return {'window_cases': cases, 'softmax_max_absolute_error': maximum,
            'compact_normalization_and_traced_rounding_exact': True,
            'note': 'CPU primitive parity only; exported full-model placement, timing and output still require verification.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    p.add_argument('--export', type=Path)
    p.add_argument('--export-pre-block', type=Path,
                   help='Diagnostic full-resolution input adapter + block 0 only; not a replacement appearance model')
    p.add_argument('--weights', type=Path, default=Path('build/teacher/logical.safetensors'))
    p.add_argument('--no-chunks', action='store_true',
                   help='Preserve spatial tensors instead of reference CPU memory-saving flatten/slice chunks')
    p.add_argument('--compact-arithmetic', action='store_true',
                   help='Preserve normalization arithmetic order while removing stacks and tensor-shaped constants')
    a = p.parse_args()
    if a.no_chunks:
        reference.CHUNK_TOKENS = 0
    result = verify()
    print(json.dumps(result, indent=2), flush=True)
    if a.export_pre_block:
        import coremltools as ct
        import numpy as np
        install_adapters(a.compact_arithmetic)
        class PreBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = reference.load_model(a.weights)
            def forward(self, x):
                value = x.permute(0, 2, 3, 1) @ self.model.weight('block0.layer0.input_adapter_weight')
                return self.model._window(value, 0, head_count=1).permute(0, 3, 1, 2)
        with torch.inference_mode():
            traced = torch.jit.trace(PreBlock().eval(), torch.zeros(1, 16, 768, 1280))
        converted = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT16, skip_model_load=True,
            inputs=[ct.TensorType(name='color', shape=(1,16,768,1280), dtype=np.float32)],
            outputs=[ct.TensorType(name='restored', dtype=np.float32)])
        converted.short_description = 'Diagnostic only: full-resolution input adapter and block0, not full neural rendering'
        converted.save(str(a.export_pre_block))
    if a.export:
        install_adapters(a.compact_arithmetic)
        from mlxdlss.tools.convert_neural_rendering_coreml import convert_weights
        convert_weights(a.weights, a.export, precision='float16', height=768, width=1280)


if __name__ == '__main__':
    main()
