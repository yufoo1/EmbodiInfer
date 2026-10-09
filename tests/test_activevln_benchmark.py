"""Replay admission and the boundary between output parity and navigation accuracy."""

from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

_comparison_module = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/activevln-benchmark/compare.py")
)
_compare = _comparison_module["compare"]


def report():
    """A synthetic complete-size report for testing the admission gate only."""
    rows = [
        {
            "sample_id": f"R2R/{i % 48 + 1}/{i}.jpg",
            "repeat": 0,
            "token_ids": [1, 2],
            "output_sha256": "same-action",
            "action_mask": [True, False, False],
            "parsed_valid": True,
            "stop_reason": "eos",
            "cache_length": i + 3,
            "generated_tokens": 2,
            "action_slots": 3,
            "latency_ms": 30.0,
        }
        for i in range(2997)
    ]
    return {
        "model": "activevln",
        "batch_size": 1,
        "dataset": {"name": "R2R"},
        "episodes": list(range(1, 49)),
        "selection_sha256": "selection",
        "tokenizer_sha256": {"tokenizer.json": "tokenizer"},
        "timing_boundary": "CPU RGB to CPU action chunk",
        "memory_protocol": "generated history; reset at episode boundary",
        "metrics": {"calls": len(rows)},
        "config": {
            "checkpoint_revision": "checkpoint",
            "dtype": "bfloat16",
            "seed": 42,
            "warmup_calls": 33,
            "repeats": 1,
            "max_new_tokens": 512,
            "max_context": 128000,
            "do_sample": False,
            "repetition_penalty": 1.05,
            "max_steps_per_episode": None,
            "cpu_threads": 4,
        },
        "rows": rows,
    }


def test_admission_requires_behavior_parity_even_when_actions_match():
    baseline = report()
    candidate = copy.deepcopy(baseline)
    assert _compare(baseline, candidate)["admitted"]
    candidate["rows"][5]["token_ids"] = [3, 2]
    result = _compare(baseline, candidate)
    assert not result["admitted"]
    assert result["matching_calls"] == 2996
    assert result["mismatches_by_field"]["token_ids"] == 1
    assert result["mismatches_by_field"]["output_sha256"] == 0


def batch_report():
    """Complete synthetic counts with a short final batch, without a model run."""
    source = report()
    selected = [{"episode_id": i + 1, "frames": 63 if i < 21 else 62} for i in range(48)]
    rows = []
    for episode in selected:
        for step in range(episode["frames"]):
            row = dict(source["rows"][len(rows)])
            row.update(
                episode_id=episode["episode_id"],
                step=step,
                batch_index=len(rows) // 4,
                actions=[[1.0, 0.0]],
                text="move forward 25cm",
            )
            rows.append(row)
    batches = [
        {
            "batch_index": index // 4,
            "observations": len(rows[index : index + 4]),
            "latency_ms": 30.0 * len(rows[index : index + 4]),
        }
        for index in range(0, len(rows), 4)
    ]
    return dict(
        source,
        schema="activevln.tensor_batch_benchmark.v2",
        status="complete",
        dataset="R2R",
        batch_size=4,
        selection=selected,
        global_episode_ids=list(range(1, 49)),
        slot_refill="ordered",
        rows=rows,
        batches=batches,
    )


def multi_instance_reports():
    """Split the complete workload once, retaining independent replica batch indices."""
    source = batch_report()
    replicas = []
    for index in range(2):
        selection = source["selection"][index::2]
        episodes = {episode["episode_id"] for episode in selection}
        rows = [copy.deepcopy(row) for row in source["rows"] if row["episode_id"] in episodes]
        for position, row in enumerate(rows):
            row["batch_index"] = position // 4
        batches = [
            {
                "batch_index": position // 4,
                "observations": len(rows[position : position + 4]),
                "latency_ms": 30.0 * len(rows[position : position + 4]),
            }
            for position in range(0, len(rows), 4)
        ]
        replicas.append(
            {
                **source,
                "config": {**source["config"], "episode_shards": 2, "episode_shard_index": index},
                "selection": selection,
                "selection_sha256": f"shard-{index}",
                "rows": rows,
                "batches": batches,
                "measurement_start_ns": 1_000_000_000,
                "measurement_end_ns": 1_000_000_000 + len(rows) * 30_000_000,
                "metrics": {"observations_per_second": 1000 / 30},
            }
        )
    return replicas


@pytest.fixture
def compare_multi(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/activevln-benchmark"))
    return _comparison_module["compare_multi_instance"]


def test_multi_instance_comparison_checks_complete_shards_without_mutating_reports(compare_multi):
    baseline = multi_instance_reports()
    candidate = copy.deepcopy(baseline)
    saved = copy.deepcopy(candidate)
    result = compare_multi(baseline, candidate)
    assert result["observations"] == 2997
    assert result["exact_behavior_parity"] and result["admitted"]
    assert result["instances"] == 2
    assert result["candidate_observations_per_second"] == pytest.approx(1000 / 30)
    assert candidate == saved
    candidate[1]["rows"][0]["token_ids"] = [99]
    changed = compare_multi(baseline, candidate)
    assert changed["mismatches_by_field"]["token_ids"] == 1
    assert not changed["admitted"]


def test_multi_instance_gate_includes_slow_tail_and_reporting_time(compare_multi):
    baseline = multi_instance_reports()
    candidate = copy.deepcopy(baseline)
    candidate[1]["measurement_end_ns"] = 91_000_000_000
    result = compare_multi(baseline, candidate)
    assert result["inference_interval_admitted"]
    assert result["candidate_wall_observations_per_second_per_gpu"] == pytest.approx(2997 / 90 / 2)
    assert not result["admitted"]


@pytest.mark.parametrize("failure", ["assignment", "missing_frame", "condition", "schedule", "incomplete"])
def test_multi_instance_comparison_rejects_changed_workload(compare_multi, failure):
    baseline = multi_instance_reports()
    candidate = copy.deepcopy(baseline)
    if failure == "assignment":
        candidate.reverse()
    elif failure == "missing_frame":
        candidate[0]["rows"].pop()
    elif failure == "condition":
        candidate[1]["config"]["max_new_tokens"] += 1
    elif failure == "schedule":
        candidate[1]["rows"][0]["batch_index"] = 1
        candidate[1]["rows"][4]["batch_index"] = 0
    else:
        candidate[1]["status"] = "error"
    with pytest.raises(ValueError):
        compare_multi(baseline, candidate)


def test_multi_instance_loader_rejects_failed_processes(tmp_path):
    with pytest.raises(ValueError, match="successful replicas"):
        _comparison_module["load_replica_reports"](
            tmp_path / "summary.json",
            {
                "status": "complete",
                "instances": 2,
                "replica_reports": ["replica-0.json", "replica-1.json"],
                "returncodes": [0, 1],
            },
        )


def test_tensor_batch_comparison_counts_real_observations_and_checks_tokens():
    baseline = batch_report()
    candidate = copy.deepcopy(baseline)
    result = _compare(baseline, candidate)
    assert result["observations"] == 2997
    assert result["candidate_amortized_e2e_ms"] == 30
    assert result["candidate_observations_per_second"] == pytest.approx(1000 / 30)
    assert result["admitted"]
    candidate["rows"][3]["token_ids"] = [9]
    result = _compare(baseline, candidate)
    assert not result["admitted"]
    assert result["mismatches_by_field"]["token_ids"] == 1


@pytest.mark.parametrize("failure", ["duplicate", "occupancy", "schedule", "partial", "latency"])
def test_tensor_batch_comparison_rejects_incomparable_reports(failure):
    baseline = batch_report()
    candidate = copy.deepcopy(baseline)
    if failure == "duplicate":
        candidate["rows"][0] = candidate["rows"][1]
    elif failure == "occupancy":
        candidate["batches"][-1]["observations"] = 4
    elif failure == "schedule":
        candidate["rows"][0], candidate["rows"][1] = candidate["rows"][1], candidate["rows"][0]
    elif failure == "partial":
        candidate["status"] = "partial_probe"
    else:
        candidate["batches"][0]["latency_ms"] = float("nan")
    with pytest.raises(ValueError):
        _compare(baseline, candidate)


def test_admission_checks_raw_latency_and_strict_target():
    baseline = report()
    candidate = copy.deepcopy(baseline)
    for row in candidate["rows"]:
        row["latency_ms"] = 40.0
    result = _compare(baseline, candidate)
    assert result["exact_behavior_parity"]
    assert not result["admitted"]


def test_task_success_contract_keeps_differences_diagnostic_and_accuracy_pending():
    baseline = report()
    candidate = copy.deepcopy(baseline)
    candidate["rows"][5]["token_ids"] = [3, 2]
    candidate["rows"][5]["output_sha256"] = "different-action"
    result = _compare(baseline, candidate, accuracy_contract="task-success")
    assert result["e2e_mean_below_40ms"]
    assert result["mismatches_by_field"]["output_sha256"] == 1
    assert result["admitted"] is None
    assert result["admission_status"] == "pending_navigation_evaluation"
    assert result["navigation_success_evaluated"] is False
    candidate["rows"].pop()
    with pytest.raises(ValueError, match="every frame"):
        _compare(baseline, candidate, accuracy_contract="task-success")


def test_comparison_rejects_unknown_accuracy_contract():
    with pytest.raises(ValueError, match="accuracy_contract"):
        _compare(report(), report(), accuracy_contract="ignore-errors")


@pytest.mark.parametrize("latency", [0, -1, float("nan"), float("inf")])
def test_admission_rejects_invalid_latency(latency):
    baseline = report()
    candidate = copy.deepcopy(baseline)
    candidate["rows"][0]["latency_ms"] = latency
    with pytest.raises(ValueError, match="finite and positive"):
        _compare(baseline, candidate)


@pytest.mark.parametrize("failure", ["truncated", "missing", "duplicate", "conditions"])
def test_admission_rejects_incomparable_measurements(failure):
    baseline = report()
    candidate = copy.deepcopy(baseline)
    if failure == "truncated":
        candidate["config"]["max_steps_per_episode"] = 64
    elif failure == "missing":
        candidate["rows"].pop()
    elif failure == "duplicate":
        candidate["rows"][1] = candidate["rows"][0]
    else:
        candidate["config"]["max_new_tokens"] = 1
    with pytest.raises(ValueError):
        _compare(baseline, candidate)
