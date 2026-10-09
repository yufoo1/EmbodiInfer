"""Aggregate throughput must measure concurrent wall time and complete unique work."""

from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

aggregate = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/activevln-benchmark/benchmark_multi_instance.py")
)["aggregate"]


def reports() -> list[dict]:
    """Two concurrent replicas with unequal durations and unequal batch occupancy."""
    return [
        {
            "status": "complete",
            "global_episode_ids": [1, 2],
            "selection": [{"episode_id": index + 1, "frames": count}],
            "rows": [{"episode_id": index + 1, "step": step} for step in range(count)],
            "measurement_start_ns": 1_000_000_000,
            "measurement_end_ns": end,
            "batches": [{"observations": count, "latency_ms": latency}],
            "metrics": {"observations_per_second": count * 1000 / latency},
        }
        for index, count, end, latency in [(0, 2, 2_000_000_000, 500), (1, 1, 3_000_000_000, 1000)]
    ]


def test_aggregate_uses_slowest_replica_wall_time_and_weighted_request_latency():
    result = aggregate(reports())
    assert result["observations_per_second"] == 1.5
    assert result["replica_inference_rates_sum"] == 5
    assert result["per_instance_amortized_ms_per_observation"] == 500
    assert result["per_instance_observations_per_second"] == 2
    assert result["request_latency_ms"]["mean"] == pytest.approx(2000 / 3)
    assert result["request_latency_ms"]["p50"] == 500
    assert result["mean_batch_occupancy"] == 1.5


@pytest.mark.parametrize("status", ["oom", "error", "partial_probe", "running", "crashed"])
def test_failed_or_partial_replica_has_no_admitted_rate(status):
    data = reports()
    data[1]["status"] = status
    assert aggregate(data) == {"status": "incomplete", "observations_per_second": None}


def test_failed_process_invalidates_an_otherwise_complete_report():
    assert aggregate(reports(), returncodes=[0, 1]) == {
        "status": "incomplete",
        "observations_per_second": None,
    }


def test_return_codes_must_cover_every_report():
    with pytest.raises(ValueError, match="one process return code"):
        aggregate(reports(), returncodes=[0])


@pytest.mark.parametrize("failure", ["duplicate_episode", "missing_frame", "duplicate_frame"])
def test_aggregate_rejects_duplicate_or_missing_work(failure):
    data = copy.deepcopy(reports())
    if failure == "duplicate_episode":
        data[1]["selection"][0]["episode_id"] = 1
    elif failure == "missing_frame":
        data[0]["rows"].pop()
    else:
        data[0]["rows"][1] = data[0]["rows"][0]
    with pytest.raises(ValueError, match="exactly once"):
        aggregate(data)
