"""Compare matched BF16 and INT8 R2R runs with the same workload and runtime switches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bf16/scripts"))

from prepare_checkpoint import digest  # noqa: E402
from summarize import audit_report  # noqa: E402

COMPARISON_KEYS = (
    "dtype",
    "batch_size",
    "seed",
    "warmup_calls",
    "repeats",
    "checkpoint_revision",
    "graph_workspace_tokens",
    "kv_pool_tokens",
    "max_new_tokens",
    "max_context",
    "do_sample",
    "repetition_penalty",
    "cpu_threads",
    "max_steps_per_episode",
    "action_space",
    "attention",
    "cuda_graph",
    "fused_ops",
    "split_attention",
    "tree_decode",
    "tree_fp32_projection",
    "grammar_tree_budget",
    "grammar_refresh",
    "grammar_prefix_weight",
    "grammar_local_weight",
    "sampling_seed",
    "gpu_preprocess",
    "uint8_observations",
    "attention_kernel",
    "incremental_kv_commit",
    "vision_kernel",
    "skip_padded_queries",
    "packed_prefill",
    "packed_prefill_bucket",
    "decode_linear",
    "query_bucket_size",
    "tree_repeat_actions",
    "prewarm_context_buckets",
)


def compare(
    reference: dict[str, Any], candidate: dict[str, Any], *, allow_probe: bool = False
) -> dict[str, Any]:
    """Reject incomplete or unmatched runs; report speed separately from output agreement."""
    statuses = {"complete", "partial_probe"} if allow_probe else {"complete"}
    for report in (reference, candidate):
        if report.get("status") not in statuses:
            raise ValueError("Comparison requires completed runs; short probes need --allow-probe")
        audit_report(report)
        if not report["rows"]:
            raise ValueError("Comparison needs measured observations")
    if reference["status"] != candidate["status"]:
        raise ValueError("Full runs and short probes cannot be compared")
    if reference.get("quantization", {}).get("enabled", False):
        raise ValueError("Reference must be unquantized")
    if candidate.get("quantization", {}).get("method") != "int8":
        raise ValueError("Candidate must report actual INT8 projections")
    differences = [
        key for key in COMPARISON_KEYS if reference["config"].get(key) != candidate["config"].get(key)
    ]
    if differences:
        raise ValueError(f"Runtime settings differ: {differences}")
    for key in ("repository", "revision", "episode_limit"):
        if reference["config"]["datasets"][0][key] != candidate["config"]["datasets"][0][key]:
            raise ValueError(f"Dataset settings differ: {key}")
    if reference["selection"] != candidate["selection"]:
        raise ValueError("Episode/frame selections differ")
    baseline = {(row["episode_id"], row["step"]): row for row in reference["rows"]}
    actual = {(row["episode_id"], row["step"]): row for row in candidate["rows"]}
    if baseline.keys() != actual.keys():
        raise ValueError("Measured observation identities differ")
    pairs = [(row, actual[key]) for key, row in baseline.items()]
    ref_time = reference["metrics"]["batch_e2e_ms"]["mean"]
    candidate_time = candidate["metrics"]["batch_e2e_ms"]["mean"]
    return {
        "status": reference["status"],
        "observations": len(pairs),
        "episodes": len(reference["selection"]),
        "bf16_e2e_ms": reference["metrics"]["batch_e2e_ms"],
        "int8_e2e_ms": candidate["metrics"]["batch_e2e_ms"],
        "e2e_speedup": ref_time / candidate_time,
        "e2e_reduction_percent": (1 - candidate_time / ref_time) * 100,
        "bf16_forward_ms": reference["metrics"]["batch_forward_ms"],
        "int8_forward_ms": candidate["metrics"]["batch_forward_ms"],
        "bf16_memory": reference["memory"],
        "int8_memory": candidate["memory"],
        "equal_token_rows": sum(a["token_ids"] == b["token_ids"] for a, b in pairs),
        "equal_action_rows": sum(a["output_sha256"] == b["output_sha256"] for a, b in pairs),
        "equal_first_action_rows": sum(a["actions"][0] == b["actions"][0] for a, b in pairs),
        "bf16_generated_tokens": sum(len(row["token_ids"]) for row in reference["rows"]),
        "int8_generated_tokens": sum(len(row["token_ids"]) for row in candidate["rows"]),
        "int8_valid_action_rows": sum(bool(row["parsed_valid"]) for row in candidate["rows"]),
        "bf16_runtime": reference.get("runtime"),
        "int8_runtime": candidate.get("runtime"),
        "bf16_environment": reference["environment"],
        "int8_environment": candidate["environment"],
        "quantization": candidate["quantization"],
        "accuracy_scope": "Independent generated histories on the same recorded observations; "
        "not teacher-forced logits or navigation SR/SPL.",
    }


def main() -> None:
    """Write a local comparison without modifying either source report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-probe", action="store_true", help="Compare completed short selections")
    args = parser.parse_args()
    result = compare(
        json.loads(args.reference.read_text()),
        json.loads(args.candidate.read_text()),
        allow_probe=args.allow_probe,
    )
    result["report_sha256"] = {"reference": digest(args.reference), "candidate": digest(args.candidate)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "observations",
                    "e2e_speedup",
                    "e2e_reduction_percent",
                    "equal_token_rows",
                    "equal_action_rows",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
