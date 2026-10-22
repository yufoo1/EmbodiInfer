"""Audit retained per-frame evidence and regenerate compact R2R result tables."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_checkpoint import digest


def percentile(values: list[float], quantile: float) -> float:
    """Match NumPy's default linear percentile interpolation without importing NumPy."""
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def audit_report(report: dict[str, Any]) -> None:
    """Reject duplicate/missing frames and inconsistent recorded timing aggregates."""
    rows = report["rows"]
    keys = [(row["episode_id"], row["step"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate episode/frame rows")
    selected = {row["episode_id"]: row["frames"] for row in report["selection"]}
    counts = Counter(row["episode_id"] for row in rows)
    if report["expected_observations"] != sum(selected.values()):
        raise ValueError("Expected observations do not match the selection")
    for episode, count in counts.items():
        steps = sorted(row["step"] for row in rows if row["episode_id"] == episode)
        if episode not in selected or count > selected[episode] or steps != list(range(count)):
            raise ValueError(f"Invalid frame sequence for episode {episode}")
    if report["status"] in ("complete", "partial_probe") and dict(counts) != selected:
        raise ValueError("Successful report is missing selected observations")
    batches = report["batches"]
    if sum(batch["observations"] for batch in batches) != len(rows):
        raise ValueError("Batch/row observation counts differ")
    if not batches:
        return
    for field, values in {
        "batch_e2e_ms": [batch["latency_ms"] for batch in batches],
        "batch_forward_ms": [batch["model_timing_ms"]["pure_inference_ms"] for batch in batches],
    }.items():
        derived = {"mean": statistics.mean(values)}
        derived.update({f"p{q}": percentile(values, q / 100) for q in (50, 95, 99)})
        for key, value in derived.items():
            if not math.isclose(value, report["metrics"][field][key], rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(f"Inconsistent {field}.{key}")
    throughput = len(rows) * 1000 / sum(batch["latency_ms"] for batch in batches)
    if not math.isclose(throughput, report["metrics"]["observations_per_second"], rel_tol=1e-12):
        raise ValueError("Inconsistent inference-interval throughput")


def summarize(root: Path) -> dict[str, Any]:
    """Keep partial/failing runs distinct and sample system RAM over the whole process."""
    provenance = root / "provenance.json"
    if provenance.exists():
        for row in json.loads(provenance.read_text())["files"]:
            if digest(root / row["file"]) != row["published_sha256"]:
                raise ValueError(f"Evidence checksum mismatch: {row['file']}")
    summaries = []
    for path in sorted((root / "reports").rglob("*.json")):
        report = json.loads(path.read_text())
        if report.get("schema") != "activevln.tensor_batch_benchmark.v2":
            continue
        audit_report(report)
        relative = path.relative_to(root / "reports")
        logs = root / "logs" / relative.parent
        telemetry = logs / (path.stem + ".tegrastats")
        text = telemetry.read_text() if telemetry.exists() else ""

        def peak(pattern: str, text: str = text) -> float | None:
            values = [float(value) for value in re.findall(pattern, text)]
            return max(values) if values else None

        rows = report["rows"]
        counts = Counter(row["episode_id"] for row in rows)
        stdout = logs / (path.stem + ".log")
        resource = logs / (path.stem + ".resource.json")
        summaries.append(
            {
                "report": str(path.relative_to(root)),
                "status": report["status"],
                "failure_phase": report["phase"] if report["status"] == "error" else None,
                "observations": len(rows),
                "expected_observations": report["expected_observations"],
                "episodes_complete": sum(
                    counts[row["episode_id"]] == row["frames"] for row in report["selection"]
                ),
                "episodes_selected": len(report["selection"]),
                "batch_size": report["batch_size"],
                "dtype": report["config"]["dtype"],
                "quantization": report.get("quantization", {"enabled": False}),
                "workspace_tokens": report["config"]["graph_workspace_tokens"],
                "warmup": report.get("warmup"),
                "metrics": report.get("metrics"),
                "gpu_memory": report.get("memory"),
                "startup_gpu_memory": report.get("startup_memory"),
                "whole_run_system_ram_peak_MiB": peak(r"RAM (\d+)/"),
                "whole_run_swap_peak_MiB": peak(r"SWAP (\d+)/"),
                "whole_run_gpu_temperature_peak_C": peak(r"gpu@([\d.]+)C"),
                "system_sampling_interval_ms": 1000,
                "max_history_tokens": max((row["cache_length"] for row in rows), default=0),
                "valid_action_rows": sum(bool(row["parsed_valid"]) for row in rows),
                "inflight_at_failure": report.get("inflight_batch"),
                "exception": report.get("exception"),
                "runtime": report.get("runtime"),
                "model_load_seconds": report.get("model_load_seconds"),
                "capture_seconds": report.get("graph_capture_seconds"),
                "warmup_seconds": report.get("warmup_seconds"),
                "measurement_wall_seconds": report.get("measurement_wall_seconds"),
                "process_resources": json.loads(resource.read_text()) if resource.exists() else None,
                "nvmap_allocation_errno12_observed": bool(
                    stdout.exists()
                    and re.search(r"NvMapMemAllocInternalTagged:.*error 12", stdout.read_text())
                ),
            }
        )
    if not summaries:
        raise ValueError("No tensor-batch reports found")
    hardware = root / "logs/hardware-settings.json"
    result = {
        "hardware_settings": json.loads(hardware.read_text()) if hardware.exists() else None,
        "scope": "Offline recorded R2R replay. Precision is recorded per run; no navigation SR/SPL claim.",
        "system_memory_scope": "Whole process including loading/capture/warmup; sampled every second. "
        "Tegrastats MB interpreted as MiB. CUDA shares physical RAM; do not sum them.",
        "runs": summaries,
    }
    old_path, new_path = root / "reports/r2r-b1-full.json", root / "reports/r2r-b1-full32k.json"
    if old_path.exists() and new_path.exists():
        old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
        lookup = {(row["episode_id"], row["step"]): row for row in new["rows"]}
        pairs = [(row, lookup[(row["episode_id"], row["step"])]) for row in old["rows"]]
        result["64k_vs_32k_overlap"] = {
            "observations": len(pairs),
            "token_ids_equal": sum(a["token_ids"] == b["token_ids"] for a, b in pairs),
            "action_hash_equal": sum(a["output_sha256"] == b["output_sha256"] for a, b in pairs),
            "scope": "Only the observed common prefix; not a serial baseline or navigation parity test.",
        }
    return result


def csv_table(summary: dict[str, Any]) -> str:
    """Render the same successful and failed runs as a compact comparison table."""
    stream = io.StringIO(newline="")
    fields = [
        "report",
        "status",
        "observations",
        "episodes_complete",
        "workspace_tokens",
        "mean_ms",
        "p95_ms",
        "cuda_peak_allocated_GiB",
        "system_ram_peak_MiB",
        "swap_peak_MiB",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in summary["runs"]:
        values = {key: row[key] for key in fields[:5]}
        timing = row["metrics"].get("batch_e2e_ms", {})
        values.update(
            mean_ms=timing.get("mean"),
            p95_ms=timing.get("p95"),
            cuda_peak_allocated_GiB=row["gpu_memory"]["peak_allocated_bytes"] / 1024**3,
            system_ram_peak_MiB=row["whole_run_system_ram_peak_MiB"],
            swap_peak_MiB=row["whole_run_swap_peak_MiB"],
        )
        writer.writerow(values)
    return stream.getvalue()


def main() -> None:
    """Write summaries or check existing ones against all retained evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate without changing summary files")
    args = parser.parse_args()
    result = summarize(args.directory)
    summary_path, csv_path = args.directory / "summary.json", args.directory / "results.csv"
    table = csv_table(result)
    if args.check:
        if json.loads(summary_path.read_text()) != result or csv_path.read_text() != table:
            raise ValueError("Summary differs from the recorded evidence")
        print(f"Evidence verified: {len(result['runs'])} runs")
    else:
        summary_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        csv_path.write_text(table)
        print(table, end="")


if __name__ == "__main__":
    main()
