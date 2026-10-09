"""FP8 activation conversion and native scale-layout regressions."""

from __future__ import annotations

import pytest
import torch

from embodiinfer.backend.torch import fp8


@pytest.mark.parametrize("tensorwise", [False, True])
def test_native_fp8_matches_activation_and_weight_scale_layout(monkeypatch, tensorwise):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    x = torch.randn(3, 16)
    weight = torch.randn(32, 16).to(torch.float8_e4m3fn)
    scale = torch.ones(()) if tensorwise else torch.ones(32)
    seen = {}

    def scaled(a, b, *, scale_a, scale_b, bias, out_dtype):
        seen.update(a=scale_a.shape, b=scale_b.shape)
        return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", scaled)
    assert fp8.native_linear(x, weight, scale).shape == (3, 32)
    expected = (
        {"a": torch.Size([]), "b": torch.Size([])}
        if tensorwise
        else {
            "a": torch.Size([3, 1]),
            "b": torch.Size([1, 32]),
        }
    )
    assert seen == expected


@pytest.mark.gpu
@pytest.mark.parametrize("width", [65, 2048, 16384])
@pytest.mark.parametrize("tensorwise", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_fp8_conversion_matches_torch_bytes(width, tensorwise, dtype):
    from embodiinfer.backend.triton.fp8_activation import quantize_activation_rows, quantize_activation_tensor

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native E4M3 conversion requires SM89+")
    generator = torch.Generator(device="cuda").manual_seed(42)
    storage = torch.randn(7, width * 2, generator=generator, device="cuda", dtype=dtype)
    x = storage[:, ::2]
    x[0].zero_()
    x[1].fill_(1)
    x[2].mul_(1e-4)
    x[3].mul_(1e4)
    actual = quantize_activation_tensor(x) if tensorwise else quantize_activation_rows(x)
    assert actual is not None
    if tensorwise:
        maximum = x.float().abs().amax()
        scale = torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum))
        expected = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale
    else:
        expected = fp8._quantize_activation_rows(x)
    torch.testing.assert_close(actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.gpu
def test_native_fp8_graph_replay_uses_new_activations():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native W8A8 requires SM89+")
    x = torch.randn(50, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    weight, scale = fp8.quantize_weight(w, scaling_scheme="channelwise")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fp8.native_linear(x, weight, scale)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = fp8.native_linear(x, weight, scale)
    x.normal_()
    graph.replay()
    quantized, activation_scale = fp8._quantize_activation_rows(x)
    expected = torch._scaled_mm(
        quantized, weight.t(), scale_a=activation_scale, scale_b=scale[None], out_dtype=x.dtype
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
