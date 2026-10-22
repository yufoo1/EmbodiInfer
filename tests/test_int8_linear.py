"""INT8 projection contracts and integer-reference CUDA Graph checks."""

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.backend.torch.int8 import quantize_weight
from embodiinfer.backend.triton.int8 import int8_capability, int8_linear
from embodiinfer.layers.linear import INT8Config, parse_int8_config
from embodiinfer.models.linear import INT8Linear


def test_triton_int8_config_and_cpu_capability():
    assert parse_int8_config({"method": "int8", "backend": "triton"}) == INT8Config(backend="triton")
    available, reason = int8_capability(torch.zeros(2, 4), torch.zeros(4, 8, dtype=torch.int8), torch.ones(8))
    assert not available and "CUDA" in reason
    with pytest.raises(RuntimeError, match="CUDA"):
        int8_linear(torch.zeros(2, 4), torch.zeros(4, 8, dtype=torch.int8), torch.ones(8))


def _reference(inputs, weight, weight_scale, bias):
    rows = inputs.reshape(-1, inputs.shape[-1]).float()
    maximum = rows.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127.0, 1.0)
    quantized = (rows / scale).round().clamp(-127, 127).to(torch.int8)
    depth, width = weight.shape
    padded_k = (-depth) % 8
    padded_n = (-width) % 8
    product = torch._int_mm(
        F.pad(quantized, (0, padded_k, 0, max(0, 17 - len(rows)))),
        F.pad(weight.contiguous(), (0, padded_n, 0, padded_k)),
    )[: len(rows), :width].float()
    result = product * scale * weight_scale
    if bias is not None:
        result = result + bias.float()
    return result.to(inputs.dtype).reshape(*inputs.shape[:-1], width)


@pytest.mark.gpu
@pytest.mark.parametrize(("rows", "depth", "width"), [(1, 64, 128), (31, 129, 135), (128, 2048, 2048)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_int8_matches_integer_reference_with_bias_and_zero_rows(rows, depth, width, dtype):
    torch.manual_seed(42)
    inputs = torch.randn(2, rows, depth, device="cuda", dtype=dtype)
    inputs[0, 0].zero_()
    weight, scale = quantize_weight(torch.randn(width, depth, device="cuda", dtype=dtype) * 0.02)
    bias = torch.randn(width, device="cuda", dtype=dtype)
    result = int8_linear(inputs, weight, scale, bias)
    reference = _reference(inputs, weight, scale, bias)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.gpu
@torch.inference_mode()
def test_int8_strided_packed_weights_and_graph_replay():
    torch.manual_seed(18)
    linear = torch.nn.Linear(64, 512, device="cuda", dtype=torch.bfloat16).eval()
    module = INT8Linear.from_linear(linear, INT8Config(backend="triton"))
    weight, scale, bias = module.weight[:, 128:384], module.weight_scale[128:384], module.bias[128:384]
    inputs = torch.randn(1, 31, 64, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            int8_linear(inputs, weight, scale, bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = int8_linear(inputs, weight, scale, bias)
    for _ in range(2):
        inputs.normal_()
        graph.replay()
        torch.testing.assert_close(output, _reference(inputs, weight, scale, bias), rtol=0, atol=0)


@pytest.mark.gpu
def test_int8_rejects_autograd_and_incompatible_shapes():
    inputs = torch.ones(2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(64, 128, device="cuda", dtype=torch.int8)
    scales = torch.ones(128, device="cuda")
    with pytest.raises(RuntimeError, match="inference only"):
        int8_linear(inputs, weight, scales)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="expects"):
            int8_linear(inputs[:, :32], weight, scales)
        with pytest.raises(ValueError, match="bias"):
            int8_linear(inputs, weight, scales, torch.zeros(127, device="cuda"))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_channel_scaling_matches_eager_and_graph_replay(dtype):
    torch.manual_seed(7)
    inputs = torch.randn(2, 31, 129, device="cuda", dtype=dtype)
    weight, scale = quantize_weight(torch.randn(135, 129, device="cuda", dtype=dtype) * 0.02)
    channel_scale = torch.randn(129, device="cuda").exp().contiguous()
    bias = torch.randn(135, device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            int8_linear(inputs, weight, scale, bias, input_scale=channel_scale)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = int8_linear(inputs, weight, scale, bias, input_scale=channel_scale)
    for _ in range(2):
        inputs.normal_()
        inputs[0, 0].zero_()
        channel_scale.copy_(torch.randn_like(channel_scale).exp())
        graph.replay()
        balanced = (inputs.float() / channel_scale).to(dtype)
        torch.testing.assert_close(output, _reference(balanced, weight, scale, bias), rtol=0, atol=0)
    with pytest.raises(ValueError, match="input_scale"):
        int8_linear(inputs, weight, scale, input_scale=channel_scale[:-1])
