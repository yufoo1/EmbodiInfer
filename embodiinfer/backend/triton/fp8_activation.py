"""Graph-safe dynamic rowwise E4M3 activation conversion for native W8A8 GEMM."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fp8_intermediate(values):
        half = values.to(tl.float16, fp_downcast_rounding="rtz")
        sticky = (half.to(tl.float32) != values).to(tl.uint16)
        return (half.to(tl.uint16, bitcast=True) | sticky).to(tl.float16, bitcast=True)

    @triton.jit
    def _quantize_rows(inputs, output, scales, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        values = tl.load(inputs + row * WIDTH + columns, columns < WIDTH, other=0).to(tl.float32)
        maximum = tl.max(tl.abs(values), axis=0)
        # Torch scalar division multiplies by the FP32 reciprocal. Retain that
        # rounding before the tensor/tensor division and E4M3 conversion.
        scale = tl.where(maximum > 0, maximum * (1.0 / 448.0), 1.0)
        quantized = tl.minimum(tl.maximum(tl.div_rn(values, scale), -448.0), 448.0)
        # Ada converts through FP16. Round-to-odd in the intermediate format
        # prevents a truncated FP32 value landing on a false E4M3 midpoint.
        half = _fp8_intermediate(quantized)
        tl.store(output + row * WIDTH + columns, half, columns < WIDTH)
        tl.store(scales + row, scale)

    @triton.jit
    def _tensor_amax(inputs, partials, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(inputs + offsets, offsets < N, other=0).to(tl.float32)
        tl.store(partials + tl.program_id(0), tl.max(tl.abs(values), 0))

    @triton.jit
    def _tensor_scale(partials, scale, N: tl.constexpr, BLOCK: tl.constexpr):
        indices = tl.arange(0, BLOCK)
        maximum = tl.max(tl.load(partials + indices, indices < N, other=0), 0)
        tl.store(scale, tl.where(maximum > 0, maximum * (1.0 / 448.0), 1.0))

    @triton.jit
    def _quantize_tensor(inputs, output, scale, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(inputs + offsets, offsets < N, other=0).to(tl.float32)
        # A scalar Tensor scale follows Torch's scalar type-promotion rules:
        # division and clamp produce the activation dtype before conversion.
        divisor = tl.load(scale).to(inputs.dtype.element_ty).to(tl.float32)
        normalized = tl.div_rn(values, divisor).to(inputs.dtype.element_ty).to(tl.float32)
        normalized = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
        tl.store(output + offsets, _fp8_intermediate(normalized), offsets < N)


def quantize_activation_rows(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return contiguous E4M3 rows and FP32 scales, or None for unsupported inputs.

    FP32 round-to-nearest division matches the unfused Torch conversion. Each row
    has an independent positive scale, including scale one for an all-zero row.
    """
    if (
        triton is None
        or not inputs.is_cuda
        or inputs.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or inputs.shape[-1] < 1
        or inputs.shape[-1] > 32768
    ):
        return None
    major, minor = torch.cuda.get_device_capability(inputs.device)
    if major * 10 + minor < 89:
        return None
    rows = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    output = torch.empty(rows.shape, device=rows.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows.shape[0], 1), device=rows.device, dtype=torch.float32)
    if rows.shape[0]:
        block = triton.next_power_of_2(rows.shape[1])
        _quantize_rows[(rows.shape[0],)](
            rows, output, scales, rows.shape[1], block, num_warps=8 if block >= 8192 else 4
        )
    return output, scales


def quantize_activation_tensor(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Quantize with one global scale without full-size FP32 activation temporaries."""
    if (
        triton is None
        or not inputs.is_cuda
        or not inputs.numel()
        or inputs.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or inputs.numel() > 8192 * 32768
    ):
        return None
    major, minor = torch.cuda.get_device_capability(inputs.device)
    if major * 10 + minor < 89:
        return None
    rows = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    count = rows.numel()
    partial_count = triton.cdiv(count, 8192)
    partials = torch.empty(partial_count, device=rows.device, dtype=torch.float32)
    scale = torch.empty((), device=rows.device, dtype=torch.float32)
    output = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    _tensor_amax[(partial_count,)](rows, partials, count, 8192)
    _tensor_scale[(1,)](partials, scale, partial_count, triton.next_power_of_2(partial_count))
    _quantize_tensor[(triton.cdiv(count, 4096),)](rows, output, scale, count, 4096)
    return output, scale
