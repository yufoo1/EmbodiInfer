"""Pinned CUDA reduction parity and unsupported-plan fallback."""

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from embodiinfer.backend.torch.cublas_reference import ReferenceBatchLinear


def test_reference_projection_rejects_cpu_before_loading_native_library():
    with pytest.raises(ValueError, match="CUDA device"):
        ReferenceBatchLinear(torch.device("cpu"))


def _require_profile():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if (
        torch.__version__.split("+")[0] != "2.10.0"
        or torch.version.cuda != "12.8"
        or version("triton") != "3.6.0"
        or version("nvidia-cublas-cu12") != "12.8.4.1"
        or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090"
    ):
        pytest.skip("requires the pinned serial-reference CUDA profile")


@pytest.mark.gpu
@pytest.mark.parametrize("batch,query", [(1, 16), (2, 16), (3, 16), (4, 16), (4, 247)])
@torch.inference_mode()
def test_reference_norm_matches_actual_serial_torch_calls(batch, query):
    _require_profile()
    from embodiinfer.backend.triton.reference_norm import reference_rms_norm
    from embodiinfer.backend.triton.rounded_ops import rounded_rms_norm

    generator = torch.Generator(device="cuda").manual_seed(42)
    inputs = torch.randn(batch, query, 2048, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16, generator=generator)
    rows = batch if query == 16 else batch * query
    expected = (
        torch.stack([rounded_rms_norm(inputs[:, i].contiguous(), weight, 1e-6) for i in range(query)], 1)
        if query == 16
        else rounded_rms_norm(inputs, weight, 1e-6)
    )
    actual = reference_rms_norm(inputs, weight, 1e-6, reference_rows=rows)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "outputs,width,bias,tuning",
    [
        (2048, 2048, True, (21, 11, 20)),
        (256, 2048, True, (21, 11, 14)),
        (11008, 2048, False, (23, 18, 15)),
        (2048, 11008, False, (21, 11, 20)),
    ],
)
@torch.inference_mode()
def test_reference_projection_matches_serial_with_private_plan(outputs, width, bias, tuning):
    _require_profile()
    generator = torch.Generator(device="cuda").manual_seed(42)
    inputs = torch.randn(4, 16, width, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(outputs, width, device="cuda", dtype=torch.bfloat16, generator=generator)
    bias = torch.randn(outputs, device="cuda", dtype=torch.bfloat16, generator=generator) if bias else None
    backend = ReferenceBatchLinear(inputs.device)
    actual = backend.project(inputs, weight, bias, tuning=tuning)
    expected = torch.stack([F.linear(inputs[:, i].contiguous(), weight, bias) for i in range(16)], 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert backend.serial_fallback_shapes == 0


@pytest.mark.gpu
@torch.inference_mode()
def test_unsupported_projection_falls_back_once_without_changing_output(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from embodiinfer.backend.torch import cublas_reference

    calls = []
    native = SimpleNamespace(
        reference_plan_create=lambda *args: 1,
        reference_plan_execute=lambda *args: calls.append(args) or 15,
        reference_plan_destroy=lambda pointer: None,
    )
    monkeypatch.setattr(cublas_reference, "_load_library", lambda: native)
    inputs = torch.randn(1, 16, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    backend = ReferenceBatchLinear(inputs.device)
    expected = torch.stack([F.linear(inputs[:, i].contiguous(), weight) for i in range(16)], 1)
    for _ in range(2):
        torch.testing.assert_close(backend.project(inputs, weight), expected, rtol=0, atol=0)
    assert len(calls) == backend.serial_fallback_shapes == 1
