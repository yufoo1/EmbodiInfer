"""Policy-local FP8 selection and incompatible numerical profiles."""

import pytest
import torch

from embodiinfer.models.linear import FP8Linear
from embodiinfer.policies.activevln.cuda_graph import ActiveVLNGraphRuntime
from embodiinfer.policies.activevln.modeling_activevln import ActiveVLNPolicy
from embodiinfer.policies.activevln.serial_draft import validate_serial_draft_profile
from test_activevln import _Processor, _Qwen


def test_fp8_selective_text_conversion_preserves_other_parameters_and_forward():
    torch.manual_seed(7)
    qwen = _Qwen()
    head = qwen.lm_head.weight.detach().clone()
    attention = qwen.model.layers[0].self_attn.q_proj.weight.detach().clone()
    policy = ActiveVLNPolicy(
        qwen,
        _Processor(),
        quantization={"ignored_layers": ["text.layers.*.self_attn.*"], "scaling_scheme": "tensorwise"},
    ).eval()
    assert len(policy.quantized_layers) == 6
    assert all(".mlp." in name for name in policy.quantized_layers)
    torch.testing.assert_close(qwen.lm_head.weight, head, rtol=0, atol=0)
    torch.testing.assert_close(qwen.model.layers[0].self_attn.q_proj.weight, attention, rtol=0, atol=0)
    linear = qwen.model.layers[0].mlp.gate_proj
    assert isinstance(linear, FP8Linear)
    scale = linear.weight_scale.clone()
    policy.to(dtype=torch.bfloat16)
    assert linear.weight.dtype == torch.uint8
    assert linear.weight_scale.dtype == torch.float32
    torch.testing.assert_close(linear.weight_scale, scale, rtol=0, atol=0)
    x = torch.randn(1, 4, 12, dtype=torch.bfloat16)
    with torch.inference_mode():
        hidden, cache = policy._forward_chunk(x, torch.arange(4)[None, None].expand(3, 1, -1), None)
    assert torch.isfinite(hidden).all() and len(cache) == 2
    evidence = policy.quantization_stats()
    assert evidence["method"] == "fp8"
    assert len(evidence["layers"]) == 6
    assert all(layer["backend"] == "torch" for layer in evidence["layers"].values())


def test_fp8_rejects_bf16_only_verification_before_workspace_allocation():
    policy = ActiveVLNPolicy(_Qwen(), _Processor(), quantization="fp8").eval()
    with pytest.raises(ValueError, match="BF16 text projections"):
        validate_serial_draft_profile(policy, None, 4)
    with pytest.raises(ValueError, match="incompatible with FP8"):
        ActiveVLNGraphRuntime(policy, tree_decode=True, tree_fp32_projection=True)


def test_default_policy_remains_unquantized():
    policy = ActiveVLNPolicy(_Qwen(), _Processor())
    assert policy.quantized_layers == ()
    assert policy.quantization_stats() == {"method": None, "layers": {}}
