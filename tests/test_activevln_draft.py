"""Learned proposals cannot replace full-vocabulary target decisions or committed KV."""

from __future__ import annotations

import pytest
import torch

from embodiinfer.exceptions import SessionCancelledError
from embodiinfer.policies.activevln.draft_activevln import ActiveVLNDraft
from test_activevln_batching import _observation, _policy


def test_draft_checkpoint_preserves_logits_and_unknown_roots(tmp_path):
    model = ActiveVLNDraft(12, [2, 5, 9], width=16, block_size=4).eval()
    hidden, roots = torch.randn(3, 12), torch.tensor([5, 31, -1])
    expected = model.propose(hidden, roots)
    assert torch.equal(expected[:, 0], roots)
    assert set(expected[:, 1:].flatten().tolist()) <= {2, 5, 9}
    path = tmp_path / "draft.pt"
    torch.save(model.checkpoint({"seed": 42}), path)
    restored, metadata = ActiveVLNDraft.load(path)
    torch.testing.assert_close(restored(hidden, roots), model(hidden, roots), rtol=0, atol=0)
    assert metadata == {"seed": 42}
    torch.save({"schema": "wrong"}, path)
    with pytest.raises(ValueError, match="schema|checkpoint"):
        ActiveVLNDraft.load(path)


@pytest.mark.parametrize("pooled", [False, True])
@torch.inference_mode()
def test_untrained_draft_preserves_greedy_tokens_scores_and_private_histories(pooled):
    policy = _policy()
    policy.max_new_tokens = 9
    policy.repetition_penalty = 1.05
    model = ActiveVLNDraft(12, list(range(32)), width=16, block_size=4).eval()
    runtime = policy.create_batched_runtime(
        batch_size=4,
        workspace_tokens=64,
        query_bucket_size=1,
        kv_pool_tokens=256 if pooled else None,
        draft=model,
    )
    observations = [_observation("3 4"), _observation("6"), _observation("7 8 9")]
    references = [
        policy.decoder.generate_tokens(policy.encode_prefix(policy.collate([obs], [str(i)])))
        for i, obs in enumerate(observations)
    ]
    results = runtime.generate(runtime.prefill(runtime.prepare(observations)))
    for actual, expected in zip(results, references, strict=True):
        assert torch.equal(actual.token_ids, expected.token_ids)
        assert actual.stop_reason == expected.stop_reason
        torch.testing.assert_close(actual.token_logprobs, expected.token_logprobs, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.memory.packed_kv, expected.memory.packed_kv, rtol=1e-5, atol=1e-6)
    memories = [result.memory for result in results]
    snapshots = [memory.packed_kv.clone() for memory in memories]
    prefix = runtime.prefill(runtime.prepare(observations, memories))
    checks = 0

    def cancel_after_forward():
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(SessionCancelledError):
        runtime.generate(prefix, cancelled=cancel_after_forward)
    for memory, snapshot in zip(memories, snapshots, strict=True):
        torch.testing.assert_close(memory.packed_kv, snapshot, rtol=0, atol=0)
    assert runtime.counters["draft_proposed_tokens"] > runtime.counters["draft_accepted_tokens"]


@torch.inference_mode()
def test_draft_near_context_limit_uses_single_token_fallback():
    policy = _policy()
    policy.max_context = 6
    policy.max_new_tokens = 2
    model = ActiveVLNDraft(12, list(range(32)), width=16, block_size=8).eval()
    runtime = policy.create_batched_runtime(
        batch_size=1, workspace_tokens=6, query_bucket_size=1, draft=model
    )
    observation = _observation("3 4 6 7")
    expected = policy.decoder.generate_tokens(policy.encode_prefix(policy.collate([observation], ["a"])))
    result = runtime.generate(runtime.prefill(runtime.prepare([observation])))[0]
    assert torch.equal(result.token_ids, expected.token_ids)
    assert result.memory.seq_len <= 6
    assert runtime.counters["draft_proposed_tokens"] == 0


@torch.inference_mode()
def test_draft_acceptance_preserves_stop_eos_rejection_and_each_row_budget():
    from types import MethodType

    policy = _policy()
    model = ActiveVLNDraft(12, list(range(32)), width=16, block_size=4).eval()
    runtime = policy.create_batched_runtime(
        batch_size=4, workspace_tokens=64, query_bucket_size=1, draft=model
    )
    scripts = ([3, 4, 2], [5], [9, 10, 2], [6, 7, 8, 9, 10, 2])

    class PositionHead(torch.nn.Module):
        def forward(self, hidden):
            scores = hidden.new_full((*hidden.shape[:-1], 32), -10)
            for row, script in enumerate(scripts):
                index = (hidden[row, ..., 0].long() + 1).clamp(0, len(script) - 1)
                scores[row].scatter_(-1, torch.tensor(script)[index][..., None], 10)
            return scores

    def forward(self, hidden, positions, past, *, ancestors=None, write_lengths=None):
        del ancestors
        result = torch.zeros_like(hidden)
        result[..., 0] = positions[0] - 2
        for row, count in enumerate(write_lengths):
            if count:
                self._row_kv(row, past[row], past[row] + count).copy_(positions[0, row, :count, None])
        return result

    def propose(self, hidden, roots):
        del self, hidden
        suffix = torch.tensor([[4, 2, 31], [31, 31, 31], [31, 31, 31], [7, 8, 9]])
        return torch.cat((roots[:, None], suffix), dim=1)

    policy.qwen.lm_head = PositionHead()
    runtime._forward = MethodType(forward, runtime)
    model.propose = MethodType(propose, model)
    results = runtime.generate(runtime.prefill(runtime.prepare([_observation("1 1")] * 4)))
    assert [g.token_ids.tolist() for g in results] == [[[3, 4, 2]], [[5]], [[9, 10, 2]], [[6, 7, 8, 9]]]
    assert [g.stop_reason for g in results] == ["eos", "stop", "eos", "max_tokens"]
    assert runtime.counters["draft_accepted_tokens"] == 5
    for result in results:
        expected = torch.arange(result.memory.seq_len)[:, None].expand(result.memory.seq_len, 6)
        torch.testing.assert_close(result.memory.packed_kv[0, 0, 0, 0], expected.float(), rtol=0, atol=0)


@torch.inference_mode()
def test_draft_rejects_incompatible_shapes_and_static_tree_combination():
    policy = _policy()
    for model, options in [
        (ActiveVLNDraft(13, [1, 2], width=8), {}),
        (ActiveVLNDraft(12, [1, 32], width=8), {}),
        (ActiveVLNDraft(12, [1, 2], width=8), {"tree_decode": True}),
    ]:
        with pytest.raises(ValueError, match="draft"):
            policy.create_batched_runtime(batch_size=1, workspace_tokens=64, draft=model, **options)
