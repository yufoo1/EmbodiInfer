"""Opt-in ActiveVLN text quantization preserves BF16 defaults and exclusions."""

import runpy
import sys
from pathlib import Path

import pytest
import torch

from embodiinfer.models.linear import INT8Linear
from embodiinfer.policies.activevln.modeling_activevln import ActiveVLNPolicy
from test_activevln import _Layer, _Qwen
from test_activevln_batching import _Processor


def test_quantization_is_opt_in_and_keeps_attention_when_excluded():
    reference = ActiveVLNPolicy(_Qwen(), _Processor(), do_sample=False).eval()
    assert reference.quantization_stats()["layer_count"] == 0
    policy = ActiveVLNPolicy(
        _Qwen(),
        _Processor(),
        do_sample=False,
        quantization={"method": "int8", "backend": "torch", "ignored_layers": ["text.layers.*.self_attn.*"]},
    ).eval()
    for layer in policy._text.layers:
        assert isinstance(layer.mlp.gate_proj, INT8Linear)
        assert isinstance(layer.mlp.up_proj, INT8Linear)
        assert isinstance(layer.mlp.down_proj, INT8Linear)
        assert isinstance(layer.self_attn.q_proj, torch.nn.Linear)
    assert policy.quantization_stats()["layer_count"] == len(policy._text.layers) * 3
    assert policy._text.embed_tokens.weight.is_floating_point()
    assert policy._lm_head.weight.is_floating_point()
    policy.to(dtype=torch.bfloat16)
    assert all(layer.mlp.gate_proj.weight_scale.dtype == torch.float32 for layer in policy._text.layers)
    assert (
        policy._text.layers[0].mlp.gate_proj(torch.ones(1, 2, 12, dtype=torch.bfloat16)).dtype
        == torch.bfloat16
    )


def test_int8_rejects_fp32_tree_projection():
    policy = ActiveVLNPolicy(_Qwen(), _Processor(), do_sample=False, quantization="int8").eval()
    with pytest.raises(ValueError, match="does not support INT8"):
        policy.create_batched_runtime(
            batch_size=1, workspace_tokens=64, tree_decode=True, tree_fp32_projection=True
        )


def test_visual_quantization_is_independent_and_preserves_bias_and_exclusions():
    qwen = _Qwen()
    qwen.visual = torch.nn.Module()
    qwen.visual.blocks = torch.nn.ModuleList([_Layer()])
    qwen.visual.merger = torch.nn.Sequential(torch.nn.Linear(12, 12))
    qwen.visual.blocks[0].mlp.up_proj = torch.nn.Linear(12, 24, bias=True)
    bias = qwen.visual.blocks[0].mlp.up_proj.bias
    policy = ActiveVLNPolicy(
        qwen,
        _Processor(),
        do_sample=False,
        vision_quantization={
            "method": "int8",
            "backend": "torch",
            "ignored_layers": ["visual.blocks.*.self_attn.*", "visual.merger.*"],
        },
    ).eval()
    assert isinstance(policy._text.layers[0].mlp.up_proj, torch.nn.Linear)
    assert isinstance(policy._visual.blocks[0].mlp.up_proj, INT8Linear)
    assert policy._visual.blocks[0].mlp.up_proj.bias is bias
    assert isinstance(policy._visual.blocks[0].self_attn.q_proj, torch.nn.Linear)
    assert isinstance(policy._visual.merger[0], torch.nn.Linear)
    stats = policy.quantization_stats()
    assert stats["layer_count"] == 3
    assert all(layer["name"].startswith("visual.") for layer in stats["layers"])
    policy.to(dtype=torch.bfloat16)
    assert policy._visual.blocks[0].mlp.up_proj.weight_scale.dtype == torch.float32


def test_calibration_selects_block_mlps_and_preserves_visual_merger(monkeypatch):
    script = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/activevln-benchmark/orin-nx/int8/scripts/precision.py"
    )
    with monkeypatch.context() as patch:
        patch.setattr(sys, "path", list(sys.path))
        patch.syspath_prepend(str(script.parent))
        patch.delitem(sys.modules, "benchmark", raising=False)
        selected = runpy.run_path(str(script))["_selected"]
        qwen = _Qwen()
        qwen.visual = torch.nn.Module()
        qwen.visual.blocks = torch.nn.ModuleList([_Layer()])
        qwen.visual.merger = torch.nn.Module()
        qwen.visual.merger.mlp = torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.Linear(24, 12))
        policy = ActiveVLNPolicy(qwen, _Processor(), do_sample=False).eval()
        names = [name for name, _, _, _ in selected(policy)]
        assert len(names) == 9
        assert sum(name.startswith("visual.blocks.") for name in names) == 3
        assert not any(name.startswith("visual.merger.") for name in names)
