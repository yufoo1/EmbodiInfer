"""RMSNorm retaining the tested Torch 2.10 FP32 reduction order at width 2048."""

from __future__ import annotations

import torch

from .rounded_ops import _require, rounded_rms_norm

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _reference_norm(X, W, Y, EPS: tl.constexpr, THREADS: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, THREADS)
        a = tl.full((THREADS,), 0, tl.float32)
        b = tl.full((THREADS,), 0, tl.float32)
        c = tl.full((THREADS,), 0, tl.float32)
        d = tl.full((THREADS,), 0, tl.float32)
        for index in range(2048 // (THREADS * 4)):
            base = row * 2048 + (index * THREADS + lane) * 4
            x0 = tl.load(X + base).to(tl.float32)
            x1 = tl.load(X + base + 1).to(tl.float32)
            x2 = tl.load(X + base + 2).to(tl.float32)
            x3 = tl.load(X + base + 3).to(tl.float32)
            a = a + x0 * x0
            b = b + x1 * x1
            c = c + x2 * x2
            d = d + x3 * x3
        values = ((a + b) + c) + d
        reduced = tl.sum(tl.reshape(values, (THREADS // 32, 32)), axis=0)
        mean = tl.sum(reduced, axis=0) * (1.0 / 2048)
        columns = tl.arange(0, 2048)
        inputs = tl.load(X + row * 2048 + columns).to(tl.float32)
        weights = tl.load(W + columns).to(tl.float32)
        normalized = (inputs * tl.rsqrt(mean + EPS)).to(X.dtype.element_ty).to(tl.float32)
        tl.store(Y + row * 2048 + columns, normalized * weights)


def reference_rms_norm(
    inputs: torch.Tensor, weight: torch.Tensor, epsilon: float, *, reference_rows: int
) -> torch.Tensor:
    """Use a tested reference row count while retaining intermediate BF16 rounding.

    For a single reference row, use actual one-row Torch reductions. Other row
    counts reproduce the vectorized four-accumulator Torch 2.10 CUDA reduction.
    The caller owns library/hardware admission; this is not a portable guarantee.
    """
    _require(inputs, weight)
    if (
        inputs.dtype != torch.bfloat16
        or inputs.ndim < 1
        or inputs.shape[-1] != 2048
        or weight.shape != (2048,)
        or not inputs.is_contiguous()
        or not weight.is_contiguous()
        or type(reference_rows) is not int
        or reference_rows < 1
        or epsilon <= 0
    ):
        raise ValueError(
            "reference RMSNorm requires contiguous BF16 width-2048 rows and a positive row count"
        )
    if reference_rows == 1:
        return torch.cat(
            [rounded_rms_norm(row, weight, epsilon) for row in inputs.reshape(-1, 2048).split(1)], dim=0
        ).reshape_as(inputs)
    height = min(16, 2 ** (reference_rows.bit_length() - 1))
    output = torch.empty_like(inputs)
    _reference_norm[(inputs.numel() // 2048,)](
        inputs, weight, output, epsilon, 512 // height, num_warps=4, enable_fp_fusion=False
    )
    return output
