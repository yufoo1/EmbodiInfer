"""Benchmark HTTP batching must preserve row identity and commit transactionally."""

from __future__ import annotations

import io
import runpy
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from embodiinfer.engine.serve.contracts import RawImage, RawPolicyRequest
from embodiinfer.policies.activevln.execution_activevln import EXECUTION_PREFIX_SCHEMA, ExecutionPrefix


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    for name in ("set_device", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    functions = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks/activevln-benchmark/serve_batch.py")
    )

    class Runtime:
        batch_size = 2
        fail = False
        invalid = False

        def prepare(self, observations, memories):
            self.inputs = list(observations), list(memories)
            return observations

        def prefill(self, prepared):
            if self.fail:
                raise RuntimeError("injected failure")
            return prepared

        def generate(self, prefix):
            return [
                SimpleNamespace(
                    next_memory=SimpleNamespace(seq_len=int(text)),
                    actions=[torch.tensor([[float("nan") if self.invalid else int(text), 0]])],
                    traces=[
                        SimpleNamespace(
                            token_ids=torch.tensor([int(text)]),
                            text=text,
                            parsed_actions=SimpleNamespace(valid=True),
                            stop_reason="eos",
                            meta={"parsed_action_mask": torch.tensor([True])},
                        )
                    ],
                )
                for text in prefix
            ]

        def stats(self):
            return {"graph_fallbacks": 0}

    benchmark = SimpleNamespace(
        make_observation=lambda image, text: text,
        timed_model=lambda prefill, decode, device: (None, decode(prefill()), {}),
    )
    policy = SimpleNamespace(decoder=SimpleNamespace(finalize_generation=lambda generation: generation))
    return functions["TensorBatchAdapter"](
        policy, Runtime(), benchmark, torch.device("cpu"), tmp_path / "batches.jsonl"
    )


def request(slot, session, value):
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    return RawPolicyRequest(
        session,
        f"request-{session}",
        0,
        str(value),
        {"benchmark_slot": slot},
        (RawImage("observation.images.rgb", "image/png", image.getvalue()),),
        {},
    )


def test_http_arrival_order_restored_and_sessions_committed_independently(adapter):
    old = SimpleNamespace(seq_len=42)
    adapter.memories["b"] = old
    results = adapter.infer_batch([request(1, "b", 8), request(0, "a", 3)])
    assert adapter.runtime.inputs == (["3", "8"], [None, old])
    assert [result.actions[0].values["rows"] for result in results] == [[[8.0, 0.0]], [[3.0, 0.0]]]
    assert adapter.memories["a"].seq_len == 3 and adapter.memories["b"].seq_len == 8
    adapter.reset("a")
    assert list(adapter.memories) == ["b"]
    tail = adapter.infer(request(1, "b", 9))
    assert tail.actions[0].values["tensor_batch"]["observations"] == 1


@pytest.mark.parametrize("failure", ["fail", "invalid"])
def test_failed_forward_or_finalize_never_commits_any_row(adapter, failure):
    old = SimpleNamespace(seq_len=42)
    adapter.memories["a"] = old
    setattr(adapter.runtime, failure, True)
    with pytest.raises((RuntimeError, ValueError)):
        adapter.infer_batch([request(0, "a", 3), request(1, "b", 8)])
    assert adapter.memories == {"a": old}
    assert adapter.failed
    with pytest.raises(RuntimeError, match="invalidated"):
        adapter.infer(request(0, "a", 4))


def test_duplicate_sessions_or_slots_are_rejected_before_model_execution(adapter):
    with pytest.raises(ValueError, match="duplicate"):
        adapter.infer_batch([request(0, "a", 3), request(1, "a", 4)])
    with pytest.raises(ValueError, match="duplicate"):
        adapter.infer_batch([request(0, "a", 3), request(0, "b", 4)])
    assert not hasattr(adapter.runtime, "inputs")


def test_prefix_reaches_observation_and_acknowledges_exact_context(adapter):
    adapter.runtime.batch_size = 1
    adapter.execution_prefix_schema = EXECUTION_PREFIX_SCHEMA
    adapter.policy._processor = SimpleNamespace(parse_execution_prefix=ExecutionPrefix.parse)

    class ObservationText(str):
        def __new__(cls, text):
            value = super().__new__(cls, text)
            value.metadata = {}
            return value

    adapter.benchmark.make_observation = lambda image, text: ObservationText(text)
    adapter.infer(request(0, "a", 3))
    value = {"schema": EXECUTION_PREFIX_SCHEMA, "previous_step_id": 0, "remaining_actions": [[1, 12.5]]}
    next_request = replace(request(0, "a", 4), step_id=1, metadata={"activevln_execution": value})
    result = adapter.infer(next_request)
    assert adapter.runtime.inputs[0][0].metadata["activevln_execution"] == value
    assert result.actions[0].values["execution_prefix_sha256"] == ExecutionPrefix.parse(value).digest
    assert adapter.memories["a"].seq_len == 4
    assert adapter.capabilities()["execution_prefix"] == EXECUTION_PREFIX_SCHEMA
    adapter.runtime.batch_size = 2
    assert "execution_prefix" not in adapter.capabilities()


def test_prefix_rejection_never_advances_model_state(adapter):
    value = {"schema": EXECUTION_PREFIX_SCHEMA, "previous_step_id": 0, "remaining_actions": []}
    next_request = replace(request(0, "a", 4), step_id=1, metadata={"activevln_execution": value})
    with pytest.raises(ValueError, match="updated batch-one"):
        adapter.infer(next_request)
    adapter.runtime.batch_size = 1
    adapter.execution_prefix_schema = EXECUTION_PREFIX_SCHEMA
    adapter.policy._processor = SimpleNamespace(parse_execution_prefix=ExecutionPrefix.parse)
    with pytest.raises(ValueError, match="committed session"):
        adapter.infer(next_request)
    assert not adapter.memories and not adapter.failed and not hasattr(adapter.runtime, "inputs")
