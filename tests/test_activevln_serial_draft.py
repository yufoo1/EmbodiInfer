"""Serial draft admission, context audits and transactional generation."""

import pytest
import torch

from embodiinfer.exceptions import SessionCancelledError
from embodiinfer.policies.activevln.draft_activevln import ActiveVLNDraft
from embodiinfer.policies.activevln.modeling_activevln import _rmsnorm
from embodiinfer.policies.activevln.serial_draft import _first_invalid_context
from test_activevln_batching import _observation, _policy


@pytest.mark.parametrize("value", [1, "true", None])
@torch.inference_mode()
def test_serial_draft_option_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        _policy().create_batched_runtime(batch_size=4, workspace_tokens=128, serial_draft=value)


@pytest.mark.parametrize(
    "options",
    [{}, {"cuda_graph": True}, {"cuda_graph": True, "fused_ops": True}],
)
@torch.inference_mode()
def test_serial_draft_rejects_incomplete_backends_before_cuda_allocation(options):
    with pytest.raises(ValueError, match="requires CUDA graphs"):
        _policy().create_batched_runtime(batch_size=4, workspace_tokens=128, serial_draft=True, **options)


@torch.inference_mode()
def test_serial_draft_rejects_unvalidated_model_and_hardware():
    with pytest.raises(ValueError, match="validated BF16"):
        _policy().create_batched_runtime(
            batch_size=4,
            workspace_tokens=128,
            serial_draft=True,
            cuda_graph=True,
            fused_ops=True,
            split_attention=True,
        )


def test_context_audit_removes_finished_rows_and_handles_power_of_two_boundaries():
    # The longest history stops after token zero. The other row crosses 512 on
    # token two, so the exact serial sequence is 8192, 512, 1024, 1024.
    prefixes, lengths = [4096, 510], [1, 4]
    assert _first_invalid_context(prefixes, lengths, [[8192], [8192, 512, 1024, 1024]], 128000) is None
    # A draft predicting two tokens on the long row would retain its partition
    # for one step too long; that accepted suffix must be repaired.
    assert _first_invalid_context(prefixes, lengths, [[8192], [8192, 8192, 1024, 1024]], 128000) == 1
    assert _first_invalid_context(prefixes, lengths, [[8192], [8192, 512, 512, 1024]], 128000) == 2
    assert _first_invalid_context([998], [2], [[1000, 1000]], 1000) is None


class _CPUVerifier:
    """Exercise scheduling/transactions with the ordinary small Torch target."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.stop_trigger_ids = frozenset(range(32))

    def norm(self, module, inputs, *, verification):
        del verification
        return _rmsnorm(module, inputs)

    def logits(self, hidden):
        return self.runtime.policy._lm_head(hidden)

    def forward(self, hidden, positions, offsets, writes, buckets):
        del buckets
        return self.runtime._forward(hidden, positions, offsets, write_lengths=writes)


@pytest.mark.parametrize("repair", [False, True])
@torch.inference_mode()
def test_serial_draft_preserves_ragged_histories_and_cancels_before_commit(monkeypatch, repair):
    from embodiinfer.policies.activevln import serial_draft

    policy = _policy()
    policy.max_new_tokens = 9
    draft = ActiveVLNDraft(12, list(range(32)), width=16, block_size=16).eval()
    runtime = policy.create_batched_runtime(
        batch_size=4, workspace_tokens=128, query_bucket_size=1, kv_pool_tokens=512, draft=draft
    )
    observations = [_observation("3 4"), _observation("6"), _observation("7 8 9")]
    runtime.draft = None
    reference = runtime.generate(runtime.prefill(runtime.prepare(observations)))
    runtime.draft = draft
    runtime._serial_draft = _CPUVerifier(runtime)
    if repair:
        monkeypatch.setattr(serial_draft, "_first_invalid_context", lambda *args: 0)
    actual = runtime.generate(runtime.prefill(runtime.prepare(observations)))
    for left, right in zip(actual, reference):
        assert torch.equal(left.token_ids, right.token_ids)
        assert left.stop_reason == right.stop_reason
        torch.testing.assert_close(left.token_logprobs, right.token_logprobs, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(left.memory.packed_kv, right.memory.packed_kv, rtol=1e-5, atol=1e-6)
    if repair:
        assert runtime.counters["draft_schedule_repaired_batches"] == 1
        assert runtime.counters["parallel_repair_blocks"] > 0
    memories = [item.memory for item in actual]
    snapshots = [memory.packed_kv.clone() for memory in memories]
    prefix = runtime.prefill(runtime.prepare(observations, memories))
    cancelled = False

    def cancel_after_audit(*args):
        nonlocal cancelled
        cancelled = True
        return 0 if repair else None

    monkeypatch.setattr(serial_draft, "_first_invalid_context", cancel_after_audit)
    with pytest.raises(SessionCancelledError):
        runtime.generate(prefix, cancelled=lambda: cancelled)
    for memory, snapshot in zip(memories, snapshots):
        torch.testing.assert_close(memory.packed_kv, snapshot, rtol=0, atol=0)
