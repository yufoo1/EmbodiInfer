"""Independent GPU processes, episode-affine replay and measured aggregate throughput."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def save(path: Path, data: dict[str, Any]) -> None:
    """Replace a report atomically so progress readers never see partial JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def aggregate(reports: list[dict[str, Any]], *, returncodes: list[int] | None = None) -> dict[str, Any]:
    """Separate aggregate wall throughput from batch-amortized per-instance efficiency."""
    if returncodes is not None and len(returncodes) != len(reports):
        raise ValueError("each replica report must have one process return code")
    if not reports or any(report["status"] != "complete" for report in reports) or any(returncodes or []):
        return {"status": "incomplete", "observations_per_second": None}
    global_ids = reports[0]["global_episode_ids"]
    selected = [row["episode_id"] for report in reports for row in report["selection"]]
    if (
        len(set(selected)) != len(selected)
        or set(selected) != set(global_ids)
        or any(report["global_episode_ids"] != global_ids for report in reports)
    ):
        raise ValueError("replicas must cover the global episode selection exactly once")
    for report in reports:
        actual = [(row["episode_id"], row["step"]) for row in report["rows"]]
        expected = {(row["episode_id"], step) for row in report["selection"] for step in range(row["frames"])}
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("replica did not complete every selected frame exactly once")
    start = min(report["measurement_start_ns"] for report in reports)
    end = max(report["measurement_end_ns"] for report in reports)
    if end <= start:
        raise ValueError("measurement interval must be positive")
    seconds = (end - start) / 1e9
    observations = sum(len(report["rows"]) for report in reports)
    batches = [batch for report in reports for batch in report["batches"]]
    instance_ms = sum(batch["latency_ms"] for batch in batches)
    request_latencies = [batch["latency_ms"] for batch in batches for _ in range(batch["observations"])]
    ordered = sorted(request_latencies)

    def percentile(percent: int) -> float:
        position = (len(ordered) - 1) * percent / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "status": "complete",
        "observations": observations,
        "episodes": len(selected),
        "wall_seconds": seconds,
        "observations_per_second": observations / seconds,
        "aggregate_amortized_ms_per_observation": seconds * 1000 / observations,
        "per_instance_amortized_ms_per_observation": instance_ms / observations,
        "per_instance_observations_per_second": observations * 1000 / instance_ms,
        "request_latency_ms": {
            "mean": sum(request_latencies) / observations,
            **{f"p{p}": percentile(p) for p in (50, 95, 99)},
        },
        "mean_batch_occupancy": observations / len(batches),
        "start_skew_ms": (max(r["measurement_start_ns"] for r in reports) - start) / 1e6,
        "replica_inference_rates_sum": sum(r["metrics"]["observations_per_second"] for r in reports),
        "timing_boundary": "Concurrent preloaded-RGB replay, after barrier and warmup, including CPU dispatch, inference, action parsing, output auditing, periodic reporting and the slower replica tail. Excludes loading, capture, disk image decoding, HTTP and simulator time.",
    }


def worker(config_path: Path, output: Path, barrier: Path, index: int, timeout: float) -> int:
    """Load CUDA only in the GPU-isolated child and wait after complete warmup."""
    from benchmark_batch import run

    def ready() -> None:
        (barrier / f"ready-{index}").touch()
        deadline = time.monotonic() + timeout
        while not (barrier / "go").exists():
            if (barrier / "abort").exists():
                raise RuntimeError("another replica failed before the measurement barrier")
            if time.monotonic() > deadline:
                raise TimeoutError("measurement barrier timed out")
            time.sleep(0.01)

    return run(json.loads(config_path.read_text()), output, ready=ready)


def launch(config: dict[str, Any], output: Path, devices: list[str], timeout: float) -> int:
    """Run fresh processes without NCCL or inter-replica synchronization during inference."""
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be distinct physical GPU identifiers")
    output.mkdir(parents=True, exist_ok=False)
    barrier = output / "barrier"
    barrier.mkdir()
    processes = []
    logs = []
    source = Path(__file__).resolve().parents[2]
    summary: dict[str, Any] = {
        "schema": "activevln.multi_instance.v1",
        "status": "starting",
        "devices": devices,
        "instances": len(devices),
        "batch_per_instance": config["batch_size"],
        "maximum_concurrent_observations": len(devices) * config["batch_size"],
        "config": config,
        "replica_reports": [f"replica-{i}.json" for i in range(len(devices))],
    }
    save(output / "summary.json", summary)
    try:
        for index, device in enumerate(devices):
            resolved = copy.deepcopy(config)
            resolved.update(episode_shards=len(devices), episode_shard_index=index, preload_rgb=True)
            config_path = output / f"config-{index}.json"
            save(config_path, resolved)
            env = dict(os.environ)
            env.update(
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                CUDA_VISIBLE_DEVICES=device,
                PYTHONPATH=str(source),
                OMP_NUM_THREADS=str(config["cpu_threads"]),
                TOKENIZERS_PARALLELISM="false",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.7",
            )
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--config",
                str(config_path),
                "--output",
                str(output / f"replica-{index}.json"),
                "--barrier",
                str(barrier),
                "--index",
                str(index),
                "--timeout",
                str(timeout),
            ]
            log = (output / f"replica-{index}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            )
        summary["pids"] = [process.pid for process in processes]
        save(output / "summary.json", summary)
        deadline = time.monotonic() + timeout
        while not all((barrier / f"ready-{i}").exists() for i in range(len(devices))):
            if any(process.poll() is not None for process in processes):
                (barrier / "abort").touch()
                break
            if time.monotonic() > deadline:
                raise TimeoutError("replica startup timed out")
            time.sleep(0.1)
        else:
            (barrier / "go").touch()
            summary["status"] = "measuring"
            save(output / "summary.json", summary)
        while any(process.poll() is None for process in processes):
            if time.monotonic() > deadline:
                raise TimeoutError("replica run timed out")
            time.sleep(0.5)
        reports = []
        for i in range(len(processes)):
            path = output / f"replica-{i}.json"
            reports.append(json.loads(path.read_text()) if path.exists() else {"status": "crashed"})
        summary["returncodes"] = [p.returncode for p in processes]
        summary["replica_statuses"] = [r["status"] for r in reports]
        summary["metrics"] = aggregate(reports, returncodes=summary["returncodes"])
        summary["status"] = summary["metrics"]["status"]
        save(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return 0 if summary["status"] == "complete" else 2
    except BaseException as exc:
        (barrier / "abort").touch()
        summary.update(status="launcher_error", error=f"{type(exc).__name__}: {exc}")
        save(output / "summary.json", summary)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def main() -> None:
    """Accept a resolved single-dataset JSON config and an unused output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--barrier", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        code = worker(args.config, args.output, args.barrier, args.index, args.timeout)
    else:
        code = launch(
            json.loads(args.config.read_text()), args.output.resolve(), args.devices.split(","), args.timeout
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
