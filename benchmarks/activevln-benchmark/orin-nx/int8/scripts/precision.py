"""Diagnose INT8 output drift with disjoint calibration and fixed BF16 histories.

This eager-SDPA probe measures output agreement, not latency or navigation SR/SPL.
Its optional smoothing module is a calibration experiment, not a deployment kernel.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import runpy
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

BENCHMARK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCHMARK.parents[1]))
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK / "orin-nx/bf16/scripts"))

from benchmark import load_navigation, make_observation, provenance, read_rgb  # noqa: E402
from prepare_checkpoint import digest, verify  # noqa: E402

from embodiinfer.backend.torch.int8 import quantize_weight  # noqa: E402
from embodiinfer.layers.linear import INT8Config  # noqa: E402
from embodiinfer.models.linear import INT8Linear  # noqa: E402
from embodiinfer.policies.activevln.cache_activevln import ActiveVLNMemory  # noqa: E402
from embodiinfer.policies.activevln.modeling_activevln import (  # noqa: E402
    ACTIVEVLN_REVISION,
    ActiveVLNPolicy,
)
from embodiinfer.policies.activevln.processor_activevln import ActiveVLNProcessor  # noqa: E402
from embodiinfer.types import Observation  # noqa: E402

VARIANTS = {
    "text": (0.0, None),
    "text_visual": (0.0, 0.0),
    "smooth_text": (0.5, None),
    "smooth_both": (0.5, 0.5),
    "smooth_text_visual": (0.5, 0.0),
    "visual": (None, 0.0),
}


class _SmoothedLinear(INT8Linear):
    def __init__(
        self, linear: torch.nn.Linear, maximum: torch.Tensor, alpha: float, *, fused: bool = True
    ) -> None:
        weight = linear.weight.detach().float()
        scale = (
            (maximum.clamp_min(1e-5).pow(alpha) / weight.abs().amax(0).clamp_min(1e-5).pow(1 - alpha)).clamp(
                1e-5, 1e5
            )
            if alpha
            else torch.ones_like(maximum)
        )
        quantized, weight_scale = quantize_weight(weight * scale)
        super().__init__(
            quantized, weight_scale, linear.bias, compute_dtype=linear.weight.dtype, backend="triton"
        )
        self.register_buffer("input_scale", scale)
        self.alpha = alpha
        self.fused = fused

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Balance input channels in FP32, then round to BF16 before W8A8 quantization."""
        if self.alpha and not self.fused:
            inputs = (inputs.float() / self.input_scale).to(inputs.dtype)
        return super().forward(inputs)

    def _linear_metadata(self) -> dict[str, torch.Tensor]:
        return {"input_scale": self.input_scale} if self.alpha and self.fused else {}


@dataclass
class _Case:
    episode_id: int
    step: int
    observation: Observation
    memory: ActiveVLNMemory | None
    logits: torch.Tensor
    tokens: list[list[int]]


def _selected(policy: ActiveVLNPolicy) -> Iterator[tuple[str, torch.nn.Module, str, torch.nn.Linear]]:
    for scope, root in (("text", policy._text), ("visual", policy._visual)):
        for name, layer in root.named_modules():
            if (
                isinstance(layer, torch.nn.Linear)
                and ".mlp." in name
                and (scope == "text" or name.startswith("blocks."))
            ):
                yield scope + "." + name, root, name, layer


def _convert(
    policy: ActiveVLNPolicy, maxima: dict[str, torch.Tensor], variant: str, *, fused: bool = True
) -> None:
    text_alpha, visual_alpha = VARIANTS[variant]
    names = [(name, root, local) for name, root, local, _ in _selected(policy)]
    for name, root, local in names:
        alpha = text_alpha if name.startswith("text.") else visual_alpha
        if alpha is None:
            continue
        parent, attribute = local.rsplit(".", 1)
        layer = root.get_submodule(local)
        maximum = maxima[name].to(layer.weight.device)
        setattr(root.get_submodule(parent), attribute, _SmoothedLinear(layer, maximum, alpha, fused=fused))
    policy._quantization_config = INT8Config(backend="triton") if text_alpha is not None else None
    policy._vision_quantization_config = INT8Config(backend="triton") if visual_alpha is not None else None


def _load(checkpoint: Path, processor: ActiveVLNProcessor) -> ActiveVLNPolicy:
    from transformers import Qwen2_5_VLForConditionalGeneration

    def evict() -> None:
        for path in checkpoint.glob("model-*.safetensors"):
            with path.open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    evict()
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
    )
    gc.collect()
    evict()
    return ActiveVLNPolicy(
        qwen,
        processor,
        attention="sdpa",
        do_sample=False,
        max_new_tokens=64,
        max_context=8192,
    ).eval()


def _compare(policy: ActiveVLNPolicy, case: _Case) -> dict[str, Any]:
    prefix = policy.encode_prefix(policy.collate([case.observation], ["validation"]), case.memory)
    logits = prefix.next_logits.float()
    generation = policy.decoder.generate_tokens(prefix)
    actual = generation.token_ids.tolist()
    reference_logp = case.logits.double().log_softmax(-1)
    actual_logp = logits.double().log_softmax(-1)
    reference_actions = policy.parse_actions(
        policy.tokenizer.decode(case.tokens[0], skip_special_tokens=True)
    )
    actual_actions = policy.parse_actions(policy.tokenizer.decode(actual[0], skip_special_tokens=True))
    return {
        "episode_id": case.episode_id,
        "step": case.step,
        "token_equal": actual == case.tokens,
        "action_equal": actual_actions.actions == reference_actions.actions,
        "first_action_equal": actual_actions.actions[:1] == reference_actions.actions[:1],
        "parsed_valid": actual_actions.valid,
        "reference_parsed_valid": reference_actions.valid,
        "bf16_tokens": case.tokens,
        "int8_tokens": actual,
        "next_logit_rmse": float((logits - case.logits).square().mean().sqrt()),
        "next_logit_kl": float((reference_logp.exp() * (reference_logp - actual_logp)).sum()),
        "next_top1_equal": bool(logits.argmax(-1) == case.logits.argmax(-1)),
    }


def _write_replay_config(args: argparse.Namespace, episodes: tuple) -> None:
    selected_ids = {episode.episode_id for episode in episodes[1:]}
    records = json.loads((args.data_root / "annotations_v1-3.json").read_text())
    annotations = args.output_dir.resolve() / "validation-annotations.json"
    annotations.write_text(json.dumps([row for row in records if int(row["id"]) in selected_ids]))
    config = json.loads((BENCHMARK / "orin-nx/bf16/configs/r2r-b1.json").read_text())
    config.update(
        checkpoint=str(args.checkpoint.resolve()),
        output_dir=str(args.output_dir.resolve() / "reports"),
        warmup_calls=min(3, args.validation_frames),
        max_steps_per_episode=args.validation_frames,
        graph_workspace_tokens=8192,
        kv_pool_tokens=8192,
        prewarm_context_buckets=[512, 1024, 2048, 4096, 8192],
    )
    spec = config["datasets"][0]
    spec.update(
        root=str(args.data_root.resolve()), annotations=str(annotations), episode_limit=len(selected_ids)
    )
    spec.pop("selection_sha256", None)
    (args.output_dir / "replay-config.json").write_text(json.dumps(config, indent=2) + "\n")


@torch.inference_mode()
def main() -> None:
    """Calibrate one episode, then compare selected variants on separate episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--validation-episodes", type=int, default=3)
    parser.add_argument("--validation-frames", type=int, default=8)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    args = parser.parse_args()
    if min(args.calibration_frames, args.validation_episodes, args.validation_frames) < 1:
        parser.error("episode/frame counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    verify(args.checkpoint.resolve(strict=True))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    episodes = load_navigation(
        {
            "name": "R2R",
            "root": str(args.data_root),
            "annotations": str(args.data_root / "annotations_v1-3.json"),
            "episode_limit": 1 + args.validation_episodes,
        }
    )
    _write_replay_config(args, episodes)
    processor = ActiveVLNProcessor(str(args.checkpoint), ACTIVEVLN_REVISION)
    policy = _load(args.checkpoint, processor)
    maxima: dict[str, torch.Tensor] = {}

    def collect(name: str, inputs: tuple[torch.Tensor, ...]) -> None:
        value = inputs[0].reshape(-1, inputs[0].shape[-1]).float().abs().amax(0)
        maxima[name] = value if name not in maxima else torch.maximum(maxima[name], value)

    hooks = [
        layer.register_forward_pre_hook(lambda module, inputs, name=name: collect(name, inputs))
        for name, _, _, layer in _selected(policy)
    ]
    memory = None
    for step, path in enumerate(episodes[0].frames[: args.calibration_frames]):
        observation = make_observation(read_rgb(path), episodes[0].instruction)
        prefix = policy.encode_prefix(policy.collate([observation], ["calibration"]), memory)
        generation = policy.decoder.generate_tokens(prefix)
        memory = generation.memory
        print("calibration", step, flush=True)
    for hook in hooks:
        hook.remove()
    report: dict[str, Any] = {
        "status": "running",
        "scope": __doc__,
        "history": "Same BF16 pre-frame KV; generation within each frame is independent.",
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "calibration_episode_ids": [episodes[0].episode_id],
        "validation_episode_ids": [episode.episode_id for episode in episodes[1:]],
        "checkpoint_revision": ACTIVEVLN_REVISION,
        "checkpoint_manifest_sha256": digest(args.checkpoint / "conversion-manifest.json"),
        "dataset": {
            key: value
            for key, value in json.loads((BENCHMARK / "orin-nx/bf16/configs/r2r-b1.json").read_text())[
                "datasets"
            ][0].items()
            if key in ("name", "repository", "revision", "split")
        },
        "runtime": {
            "dtype": "bfloat16",
            "batch_size": 1,
            "seed": 42,
            "float32_matmul_precision": "highest",
            "attention": "sdpa",
            "cuda_graph": False,
            "max_new_tokens": 64,
            "max_context": 8192,
            "do_sample": False,
            "repetition_penalty": 1.05,
            "cpu_threads": 4,
        },
        "environment": provenance(torch.device("cuda:0")),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": [],
    }
    torch.save(
        {"maxima": {key: value.cpu() for key, value in maxima.items()}, "metadata": report},
        args.output_dir / "calibration-maxima.pt",
    )
    cases = []
    for episode in episodes[1:]:
        memory = None
        for step, path in enumerate(episode.frames[: args.validation_frames]):
            observation = make_observation(read_rgb(path), episode.instruction)
            before = memory
            prefix = policy.encode_prefix(policy.collate([observation], ["validation"]), memory)
            logits = prefix.next_logits.float().clone()
            generation = policy.decoder.generate_tokens(prefix)
            memory = generation.memory
            cases.append(
                _Case(episode.episode_id, step, observation, before, logits, generation.token_ids.tolist())
            )
            print("bf16", episode.episode_id, step, flush=True)
    del prefix, generation, memory, before, policy
    gc.collect()
    torch.cuda.empty_cache()
    for variant in args.variants:
        text_alpha, visual_alpha = VARIANTS[variant]
        policy = _load(args.checkpoint, processor)
        _convert(policy, maxima, variant)
        gc.collect()
        torch.cuda.empty_cache()
        rows = []
        for case in cases:
            row = _compare(policy, case)
            rows.append(row)
            print(variant, case.episode_id, case.step, row["token_equal"], flush=True)
        report["records"].append(
            {"variant": variant, "text_alpha": text_alpha, "visual_alpha": visual_alpha, "rows": rows}
        )
        (args.output_dir / "precision.json").write_text(json.dumps(report, indent=2) + "\n")
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "complete_probe"
    (args.output_dir / "precision.json").write_text(json.dumps(report, indent=2) + "\n")


def replay() -> None:
    """Measure a calibrated candidate through the shared, fully configured batch benchmark."""
    import benchmark_batch
    import run_with_streamed_loading

    parser = argparse.ArgumentParser(description=replay.__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unfused-smoothing", action="store_true", help="Use the eager scaling reference")
    args = parser.parse_args(sys.argv[2:])
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    if config.get("quantization") or config.get("vision_quantization"):
        raise ValueError("Use the BF16 reference config; --variant selects calibration and quantization")
    artifact = torch.load(args.calibration, map_location="cpu", weights_only=True)
    metadata = artifact.get("metadata", artifact)
    if metadata["checkpoint_revision"] != config["checkpoint_revision"]:
        raise ValueError("Calibration checkpoint revision differs")
    expected_manifest = metadata.get("checkpoint_manifest_sha256")
    if expected_manifest is not None and expected_manifest != digest(
        Path(config["checkpoint"]) / "conversion-manifest.json"
    ):
        raise ValueError("Calibration checkpoint manifest differs")
    calibration_ids = set(metadata["calibration_episode_ids"])
    episodes = load_navigation(config["datasets"][0])
    if calibration_ids.intersection(ep.episode_id for ep in episodes):
        raise ValueError("Calibration and validation episodes must be disjoint")
    config["precision_experiment"] = {
        "variant": args.variant,
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scaling": "FP32 channel division, BF16 cast, then dynamic W8A8",
        "fused_smoothing": not args.unfused_smoothing,
    }
    original_factory = benchmark_batch.make_policy
    original_run_path = runpy.run_path

    def factory(*positional: Any, **keywords: Any) -> ActiveVLNPolicy:
        policy = original_factory(*positional, **keywords)
        with torch.no_grad():
            _convert(policy, artifact["maxima"], args.variant, fused=not args.unfused_smoothing)
        return policy

    def run_path(path: str, **keywords: Any) -> Any:
        if Path(path).name != "benchmark_batch.py":
            return original_run_path(path, **keywords)
        result = benchmark_batch.run(config, args.output)
        if result:
            raise SystemExit(result)
        return {}

    benchmark_batch.make_policy = factory
    runpy.run_path = run_path
    try:
        run_with_streamed_loading.main()
    finally:
        benchmark_batch.make_policy = original_factory
        runpy.run_path = original_run_path


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        replay()
    else:
        main()
