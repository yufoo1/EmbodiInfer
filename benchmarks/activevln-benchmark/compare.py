"""Compare complete replays; navigation success requires a separate closed-loop run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

# Optimization controls may differ; the workload and generation contract may not.
CONDITIONS = (
    "checkpoint_revision",
    "dtype",
    "seed",
    "warmup_calls",
    "repeats",
    "max_new_tokens",
    "max_context",
    "do_sample",
    "repetition_penalty",
    "max_steps_per_episode",
    "cpu_threads",
)
OUTPUT_FIELDS = (
    "token_ids",
    "output_sha256",
    "action_mask",
    "parsed_valid",
    "stop_reason",
    "cache_length",
    "generated_tokens",
    "action_slots",
)
EXPECTED_CALLS = {"R2R": 2997, "RxR": 3879}


def compare_multi_instance(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    """Audit complete episode-affine shards and require 25/s per allocated GPU.

    Both runs must use the same replica assignments and batch schedules. The
    admission gate includes common wall time, including the slower replica's
    tail and reporting; summed inference intervals remain a separate metric.
    """
    from benchmark_multi_instance import aggregate

    if not reference or len(reference) != len(candidate):
        raise ValueError("comparison requires the same nonempty replica count")
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        for key in ("selection", "selection_sha256", "global_episode_ids"):
            if left[key] != right[key]:
                raise ValueError(f"replica {index} assignment changed: {key}")
        for report in (left, right):
            config = report["config"]
            if config.get("episode_shards") != len(reference) or config.get("episode_shard_index") != index:
                raise ValueError("replica index/count does not match its shard configuration")
            for key in (*CONDITIONS, "action_space", "query_bucket_size"):
                if config.get(key) != reference[0]["config"].get(key):
                    raise ValueError(f"replica experimental condition changed: {key}")
            for key in ("schema", "dataset", "batch_size", "timing_boundary", "slot_refill"):
                if report[key] != reference[0][key]:
                    raise ValueError(f"replica measurement contract changed: {key}")

    metrics = [aggregate(reports) for reports in (reference, candidate)]
    if any(result["status"] != "complete" for result in metrics):
        raise ValueError("comparison requires every replica to complete")

    def combine(reports: list[dict[str, Any]]) -> dict[str, Any]:
        combined = dict(reports[0])
        combined["selection"] = [episode for report in reports for episode in report["selection"]]
        combined["selection_sha256"] = hashlib.sha256(
            json.dumps(combined["selection"], sort_keys=True).encode()
        ).hexdigest()
        combined["rows"], combined["batches"] = [], []
        for report in reports:
            offset = len(combined["batches"])
            combined["batches"].extend(
                {**batch, "batch_index": offset + batch["batch_index"]} for batch in report["batches"]
            )
            combined["rows"].extend(
                {**row, "batch_index": offset + row["batch_index"]} for row in report["rows"]
            )
        return combined

    result = compare_batches(combine(reference), combine(candidate))
    wall_rates = [metric["observations_per_second"] / len(reference) for metric in metrics]
    result.update(
        instances=len(reference),
        baseline_wall_observations_per_second_per_gpu=wall_rates[0],
        candidate_wall_observations_per_second_per_gpu=wall_rates[1],
        wall_speedup=wall_rates[1] / wall_rates[0],
        inference_interval_admitted=result["admitted"],
        admitted=result["admitted"] and wall_rates[1] >= 25.0,
        admission_boundary="Exact output parity and at least 25 observations/s per allocated GPU over common wall time.",
    )
    return result


def load_replica_reports(summary_path: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Load complete sibling reports, rejecting missing or failed worker processes."""
    names = summary.get("replica_reports", [])
    count = summary.get("instances")
    if (
        summary.get("status") != "complete"
        or not names
        or count != len(names)
        or len(set(names)) != len(names)
        or summary.get("returncodes") != [0] * count
    ):
        raise ValueError("multi-instance summary must contain complete successful replicas")
    reports = []
    for name in names:
        if Path(name).name != name:
            raise ValueError("replica reports must be sibling filenames")
        report = json.loads((summary_path.parent / name).read_bytes())
        reports.append(report)
    return reports


def compare_batches(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Audit complete tensor-batch replays with identical occupancy and episode order.

    Rate uses actual observations divided by summed batch E2E time. Padding and
    the number of generated action slots never increase the observation count.
    """
    for key in (
        "schema",
        "dataset",
        "batch_size",
        "selection",
        "selection_sha256",
        "global_episode_ids",
        "timing_boundary",
        "slot_refill",
    ):
        if reference[key] != candidate[key]:
            raise ValueError(f"experimental condition changed: {key}")
    for key in (*CONDITIONS, "action_space", "query_bucket_size"):
        if reference["config"].get(key) != candidate["config"].get(key):
            raise ValueError(f"experimental condition changed: {key}")
    indexed, latencies, schedules = [], [], []
    for report in (reference, candidate):
        if (
            report["schema"] != "activevln.tensor_batch_benchmark.v2"
            or report["status"] != "complete"
            or report["config"]["max_steps_per_episode"] is not None
            or report["config"]["repeats"] != 1
        ):
            raise ValueError("comparison requires complete tensor-batch replays")
        selected = report["selection"]
        if len(selected) != 48 or len({row["episode_id"] for row in selected}) != 48:
            raise ValueError("comparison requires all 48 distinct selected episodes")
        expected = {(ep["episode_id"], step) for ep in selected for step in range(ep["frames"])}
        if len(expected) != EXPECTED_CALLS.get(report["dataset"]):
            raise ValueError("unexpected complete-workload observation count")
        rows = {(row["episode_id"], row["step"]): row for row in report["rows"]}
        if len(rows) != len(report["rows"]) or rows.keys() != expected:
            raise ValueError("each selected frame must occur exactly once")
        batches = report["batches"]
        schedule = [[] for _ in batches]
        for key, row in rows.items():
            index = row["batch_index"]
            if type(index) is not int or not 0 <= index < len(batches):
                raise ValueError("invalid row batch index")
            schedule[index].append(key)
        for index, (batch, members) in enumerate(zip(batches, schedule, strict=True)):
            if (
                batch["batch_index"] != index
                or batch["observations"] != len(members)
                or not 1 <= len(members) <= report["batch_size"]
                or not math.isfinite(batch["latency_ms"])
                or batch["latency_ms"] <= 0
            ):
                raise ValueError("invalid batch occupancy or latency")
        indexed.append(rows)
        latencies.append(sum(batch["latency_ms"] for batch in batches) / len(rows))
        schedules.append(schedule)
    if schedules[0] != schedules[1]:
        raise ValueError("batch membership/order changed")
    fields = (*OUTPUT_FIELDS[:6], "actions", "text")
    counts = dict.fromkeys(fields, 0)
    differences = []
    for key, row in indexed[0].items():
        changed = [field for field in fields if row[field] != indexed[1][key][field]]
        for field in changed:
            counts[field] += 1
        if changed:
            differences.append({"episode_id": key[0], "step": key[1], "fields": changed})
    return {
        "dataset": reference["dataset"],
        "observations": len(indexed[0]),
        "batch_size": reference["batch_size"],
        "exact_behavior_parity": not differences,
        "matching_observations": len(indexed[0]) - len(differences),
        "mismatches_by_field": counts,
        "first_mismatches": differences[:100],
        "baseline_amortized_e2e_ms": latencies[0],
        "candidate_amortized_e2e_ms": latencies[1],
        "candidate_observations_per_second": 1000 / latencies[1],
        "speedup": latencies[0] / latencies[1],
        "navigation_success_evaluated": False,
        "admitted": not differences and latencies[1] <= 40.0,
    }


def indexed_rows(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """Reject duplicate identities instead of silently dropping repeated rows."""
    rows = {}
    for row in report["rows"]:
        if not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
            raise ValueError("each measured latency must be finite and positive")
        key = (row["sample_id"], row["repeat"])
        if key in rows:
            raise ValueError(f"duplicate measured call: {key}")
        rows[key] = row
    return rows


def compare(
    reference: dict[str, Any], candidate: dict[str, Any], *, accuracy_contract: str = "exact"
) -> dict[str, Any]:
    """Compare full workloads under exact or externally evaluated task accuracy.

    ``task-success`` retains output differences as diagnostics, but cannot decide
    admission from recorded observations. Its admission result remains unknown
    until the two policies have been evaluated in a closed-loop simulator.
    """
    if accuracy_contract not in ("exact", "task-success"):
        raise ValueError("accuracy_contract must be exact or task-success")
    if reference.get("schema") == "activevln.tensor_batch_benchmark.v2":
        if accuracy_contract != "exact":
            raise ValueError("tensor-batch comparison currently supports only the exact contract")
        return compare_batches(reference, candidate)
    name = reference["dataset"]["name"]
    if name not in EXPECTED_CALLS or candidate["dataset"]["name"] != name:
        raise ValueError("reports must describe the same supported navigation dataset")
    for report in (reference, candidate):
        if report["model"] != "activevln" or report["batch_size"] != 1:
            raise ValueError("expected B=1 ActiveVLN reports")
        config = report["config"]
        if config["max_steps_per_episode"] is not None or config["repeats"] != 1:
            raise ValueError("admission requires one complete replay, without a frame limit")
        if len(report["episodes"]) != 48 or len(set(report["episodes"])) != 48:
            raise ValueError("admission requires 48 distinct episodes")
        if len(report["rows"]) != EXPECTED_CALLS[name]:
            raise ValueError("report does not contain every frame of the 48 selected trajectories")
        if report["metrics"]["calls"] != len(report["rows"]):
            raise ValueError("summary count does not match raw rows")
    for key in CONDITIONS:
        if reference["config"][key] != candidate["config"][key]:
            raise ValueError(f"experimental condition changed: {key}")
    for key in ("episodes", "selection_sha256", "tokenizer_sha256", "timing_boundary", "memory_protocol"):
        if reference[key] != candidate[key]:
            raise ValueError(f"measurement contract changed: {key}")
    left, right = indexed_rows(reference), indexed_rows(candidate)
    if left.keys() != right.keys():
        raise ValueError("measured call identities differ")
    differences = []
    counts = dict.fromkeys(OUTPUT_FIELDS, 0)
    for key, row in left.items():
        fields = [field for field in OUTPUT_FIELDS if row[field] != right[key][field]]
        for field in fields:
            counts[field] += 1
        if fields:
            differences.append({"sample_id": key[0], "repeat": key[1], "fields": fields})
    # Derive the target and speedup from raw timings, not potentially stale summaries.
    baseline_ms = sum(row["latency_ms"] for row in left.values()) / len(left)
    optimized_ms = sum(row["latency_ms"] for row in right.values()) / len(right)
    return {
        "dataset": name,
        "calls": len(left),
        "matching_calls": len(left) - len(differences),
        "exact_behavior_parity": not differences,
        "mismatches_by_field": counts,
        "first_mismatches": differences[:100],
        "baseline_e2e_mean_ms": baseline_ms,
        "candidate_e2e_mean_ms": optimized_ms,
        "speedup": baseline_ms / optimized_ms,
        "e2e_mean_below_40ms": optimized_ms < 40.0,
        "accuracy_contract": accuracy_contract,
        "navigation_success_evaluated": False,
        "admission_status": (
            "pending_navigation_evaluation"
            if accuracy_contract == "task-success"
            else "passed"
            if not differences and optimized_ms < 40.0
            else "failed"
        ),
        "admitted": (not differences and optimized_ms < 40.0 if accuracy_contract == "exact" else None),
    }


def main() -> None:
    """Write comparison; exit 2 when navigation accuracy still needs evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accuracy-contract",
        choices=("exact", "task-success"),
        default="exact",
        help="task-success reports output differences without treating them as accuracy failures",
    )
    args = parser.parse_args()
    reference, candidate = args.reference.read_bytes(), args.candidate.read_bytes()
    left, right = json.loads(reference), json.loads(candidate)
    if left.get("schema") == "activevln.multi_instance.v1":
        if right.get("schema") != left["schema"] or args.accuracy_contract != "exact":
            raise ValueError("multi-instance comparison requires matching summaries and exact accuracy")
        result = compare_multi_instance(
            load_replica_reports(args.reference, left), load_replica_reports(args.candidate, right)
        )
    else:
        result = compare(left, right, accuracy_contract=args.accuracy_contract)
    result["reference_sha256"] = hashlib.sha256(reference).hexdigest()
    result["candidate_sha256"] = hashlib.sha256(candidate).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(2 if result["admitted"] is None else 0 if result["admitted"] else 1)


if __name__ == "__main__":
    main()
