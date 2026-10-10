"""Opt-in verification against the pinned serial greedy ActiveVLN tensor runtime.

The owner controls private numerical plans, attention schedules and graphs.
Committed memories change only after all proposed response prefixes are valid.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from importlib.metadata import version
from itertools import combinations
from typing import TYPE_CHECKING
from weakref import proxy

import torch

from ...exceptions import SessionCancelledError
from . import speculation_activevln as speculation

if TYPE_CHECKING:
    from .batching_activevln import ActiveVLNBatchedRuntime, ActiveVLNBatchPrefix
    from .draft_activevln import ActiveVLNDraft
    from .modeling_activevln import ActiveVLNGeneration, ActiveVLNPolicy

_TensorRows = list[list[torch.Tensor]]


def validate_serial_draft_profile(
    policy: ActiveVLNPolicy, draft: ActiveVLNDraft | None, batch_size: int
) -> None:
    """Reject unvalidated numerical profiles before allocating a batched workspace."""
    if policy.quantized_layers:
        raise ValueError("serial_draft requires BF16 text projections; FP8 is a lossy profile")
    parameter = next(policy.parameters())
    shapes = ((2048, 2048), (256, 2048), (11008, 2048), (2048, 11008))
    if (
        parameter.device.type != "cuda"
        or parameter.dtype != torch.bfloat16
        or batch_size != 4
        or draft is None
        or draft.block_size != 16
        or policy.action_space != "r2r"
        or policy.max_context < 512
        or len(policy._text.layers) != 36
        or tuple(policy._lm_head.weight.shape) != (151936, 2048)
        or any(
            tuple(module.weight.shape) not in shapes
            for module in policy._text.modules()
            if isinstance(module, torch.nn.Linear)
        )
    ):
        raise ValueError("serial_draft requires the validated BF16 R2R 3B profile, B=4 and a 16-token draft")
    if (
        sys.platform != "linux"
        or torch.__version__.split("+")[0] != "2.10.0"
        or torch.get_float32_matmul_precision() != "highest"
        or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        or torch.backends.cuda.matmul.allow_fp16_accumulation
        or torch.version.cuda != "12.8"
        or version("transformers") != "4.51.3"
        or version("triton") != "3.6.0"
        or version("nvidia-cublas-cu12") != "12.8.4.1"
        or torch.cuda.get_device_name(parameter.device) != "NVIDIA GeForce RTX 4090"
    ):
        raise ValueError(
            "serial_draft requires Linux, RTX 4090, Torch 2.10.0/cu128, Transformers 4.51.3, "
            "Triton 3.6.0, cuBLAS 12.8.4.1, "
            "highest matmul precision and default CUDA reduction settings"
        )


def _first_invalid_context(
    prefix_lengths: Sequence[int], token_counts: Sequence[int], buckets: Sequence[Sequence[int]], limit: int
) -> int | None:
    """Find the first accepted token whose attention partition differs from serial.

    Finished rows no longer contribute to the serial batch's longest live KV
    prefix. Auditing actual accepted lengths catches incorrect draft EOS/STOP
    forecasts before their descendants enter committed memory.
    """
    for index in range(max(token_counts)):
        active = [row for row, count in enumerate(token_counts) if count > index]
        latest = max(prefix_lengths[row] + index for row in active)
        serial_bucket = min(limit, max(512, 1 << latest.bit_length()))
        if any(buckets[row][index] != serial_bucket for row in active):
            return index
    return None


def _check_cancel(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SessionCancelledError("batched ActiveVLN serial draft generation was cancelled")


class SerialDraftVerifier:
    """Own one runtime's serial-reference operators and per-token context graphs.

    This opt-in profile is deliberately restricted to measured numerical settings.
    It never changes model forwards or process-global attention/normalization.
    """

    def __init__(self, runtime: ActiveVLNBatchedRuntime) -> None:
        from ...backend.torch.cublas_reference import ReferenceBatchLinear

        self._runtime = proxy(runtime)
        self._linear = ReferenceBatchLinear(runtime.device)
        self.contexts: tuple[int, ...] | None = None
        self._mask = torch.zeros(runtime.batch_size, 16, device=runtime.device, dtype=torch.long)
        self._causal = torch.ones(16, 16, device=runtime.device, dtype=torch.bool).tril()
        self._causal = self._causal[None].repeat(runtime.batch_size, 1, 1)
        self._graphs: dict[tuple[int, ...], torch.cuda.CUDAGraph] = {}
        tokenizer = runtime.policy.tokenizer
        vocabulary = runtime.draft.token_ids.tolist()
        pieces = {token: tokenizer.decode([token], skip_special_tokens=True).lower() for token in vocabulary}
        safe = not getattr(tokenizer, "clean_up_tokenization_spaces", True) and all(
            piece.isascii() for piece in pieces.values()
        )
        self.stop_trigger_ids = frozenset(
            token
            for token, piece in pieces.items()
            if not safe or "stop" in piece or piece.startswith(("top", "op", "p"))
        )

    def stats(self) -> dict[str, int]:
        """Expose graph coverage and unsupported projection shapes without counters reset."""
        return {
            "serial_draft_context_graphs": len(self._graphs),
            "serial_draft_projection_fallback_shapes": self._linear.serial_fallback_shapes,
            "serial_draft_stop_trigger_tokens": len(self.stop_trigger_ids),
        }

    def invalidate(self) -> None:
        """Drop graphs before their KV workspace or shared input buffers are replaced."""
        self._graphs.clear()

    def norm(self, module: torch.nn.Module, inputs: torch.Tensor, *, verification: bool) -> torch.Tensor:
        """Preserve serial row reductions for verification, actual rows for prefill."""
        from ...backend.triton.reference_norm import reference_rms_norm

        rows = inputs.shape[0] if verification else inputs.numel() // inputs.shape[-1]
        return reference_rms_norm(inputs, module.weight, module.variance_epsilon, reference_rows=rows)

    def project(self, module: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the measured text-projection tuning with the serial batch's reduction."""
        shape = tuple(module.weight.shape)
        tuning = (
            (21, 11, 14) if shape == (256, 2048) else (23, 18, 15) if shape == (11008, 2048) else (21, 11, 20)
        )
        return self._linear.project(inputs, module.weight, module.bias, tuning=tuning)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Score the complete target vocabulary, including short final replica batches."""
        head = self._runtime.policy._lm_head
        if hidden.ndim != 3 or hidden.shape[1] != 16:
            return head(hidden)
        return self._linear.project(hidden, head.weight, head.bias)

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        offsets: torch.Tensor,
        key_bucket: int,
        *,
        cache_starts: torch.Tensor | None,
        cache_capacities: torch.Tensor | None,
    ) -> torch.Tensor:
        """Use causal attention and the serial context partition for each query row."""
        from ...backend.triton.split_attention import split_kv_attention

        contexts = (key_bucket,) if self.contexts is None else self.contexts
        options = {"cache_starts": cache_starts, "cache_capacities": cache_capacities}
        output = split_kv_attention(query, key, value, offsets, contexts[0], **options)
        for bucket in contexts[1:]:
            candidate = split_kv_attention(query, key, value, offsets, bucket, **options)
            output = torch.where((self._mask == bucket)[:, None, :, None], candidate, output)
        return output

    def prewarm(self, context_buckets: Sequence[int]) -> None:
        """Capture pairs/triples before measurement; uncommon combinations stay explicit."""
        runtime = self._runtime
        self.invalidate()
        buffers = runtime._graph_buffers[(16, True)]
        try:
            for size in (2, 3):
                for group in combinations(sorted(set(context_buckets)), size):
                    self.contexts = group
                    buffers.offsets.fill_(max(group) - 16)
                    runtime._write_lengths.zero_()

                    def forward(key_bucket: int = max(group)) -> torch.Tensor:
                        buffers.output.copy_(
                            runtime._text_forward(
                                buffers.hidden, buffers.positions, buffers.offsets, key_bucket, self._causal
                            )
                        )
                        return buffers.output

                    graph, _ = runtime._capture(forward, runtime.device, runtime.pool)
                    self._graphs[group] = graph
        finally:
            self.contexts = None
        runtime.storage.zero_()

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        offsets: list[int],
        writes: list[int],
        buckets: list[int] | list[list[int]],
    ) -> torch.Tensor:
        """Verify a private block using per-row forecast or corrected context buckets."""
        runtime = self._runtime
        if isinstance(buckets[0], list):
            group = tuple(sorted({value for row in buckets for value in row}))
        else:
            group = tuple(sorted(set(buckets)))
        runtime._write_lengths.copy_(torch.tensor(writes, device=runtime.device, dtype=torch.long))
        self._mask.copy_(torch.tensor(buckets, device=runtime.device, dtype=torch.long))
        entry = runtime.tree_graphs.get((16, group[0])) if len(group) == 1 else None
        if entry is not None:
            entry.hidden.copy_(hidden)
            entry.positions.copy_(positions)
            entry.cache_position.copy_(torch.tensor(offsets, device=runtime.device, dtype=torch.long))
            entry.graph.replay()
            return entry.output.clone()
        graph = self._graphs.get(group)
        if graph is not None:
            buffers = runtime._graph_buffers[(16, True)]
            buffers.hidden.copy_(hidden)
            buffers.positions.copy_(positions)
            buffers.offsets.copy_(torch.tensor(offsets, device=runtime.device, dtype=torch.long))
            graph.replay()
            return buffers.output.clone()
        runtime.counters["serial_draft_eager_fallbacks"] = (
            runtime.counters.get("serial_draft_eager_fallbacks", 0) + 1
        )
        self.contexts = group
        try:
            return runtime._text_forward(
                hidden,
                positions,
                torch.tensor(offsets, device=runtime.device, dtype=torch.long),
                max(group),
                self._causal,
            )
        finally:
            self.contexts = None


def _repair_serial(
    runtime: ActiveVLNBatchedRuntime,
    prefix: ActiveVLNBatchPrefix,
    trial_tokens: _TensorRows,
    trial_scores: _TensorRows,
    features: _TensorRows,
    trial_reasons: list[str],
    start: int,
    cancelled: Callable[[], bool] | None,
) -> tuple[ActiveVLNGeneration, ...]:
    """Repair an uncommitted suffix while retaining the last validated prefix."""
    policy = runtime.policy
    count = len(trial_tokens)
    device = runtime.device
    tokens = [row[:start] for row in trial_tokens]
    scores = [row[:start] for row in trial_scores]
    active = [len(row) > start for row in trial_tokens]
    reasons = ["max_tokens" if active[row] else trial_reasons[row] for row in range(count)]
    lengths = [prefix.lengths[row] + len(tokens[row]) for row in range(count)]
    coordinates = [prefix.positions[row] + len(tokens[row]) for row in range(count)]
    logits = (
        prefix.logits
        if start == 0
        else policy._lm_head(torch.stack([features[row][len(tokens[row]) - 1] for row in range(count)]))
    )
    seen = torch.zeros_like(logits, dtype=torch.bool)
    for row, history in enumerate(prefix.token_history):
        seen[row].scatter_(0, history, True)
        if tokens[row]:
            seen[row].scatter_(0, torch.stack(tokens[row]), True)
    for _ in range(start, policy.max_new_tokens):
        if cancelled is not None and cancelled():
            raise SessionCancelledError("ActiveVLN serial repair cancelled")
        if any(lengths[row] >= policy.max_context for row in range(count) if active[row]):
            raise ValueError("batched generation exceeds the model context limit")
        penalty = policy.repetition_penalty
        effective = torch.where(seen, torch.where(logits < 0, logits * penalty, logits / penalty), logits)
        token = effective.argmax(-1, keepdim=True)
        logprob = torch.log_softmax(effective, -1).gather(1, token)
        seen.scatter_(1, token, True)
        ids = torch.zeros(runtime.batch_size, 1, device=device, dtype=torch.long)
        ids[:count] = token
        position = torch.zeros(3, runtime.batch_size, 1, device=device, dtype=torch.long)
        for row in range(count):
            position[:, row, 0] = coordinates[row] if active[row] else 0
        hidden = runtime._forward(
            policy._text.embed_tokens(ids),
            position,
            [lengths[row] if active[row] else 0 for row in range(count)] + [0] * (runtime.batch_size - count),
            write_lengths=[int(v) for v in active] + [0] * (runtime.batch_size - count),
        )
        logits = policy._lm_head(hidden[:count, -1])
        values = token[:, 0].tolist()
        runtime.counters["draft_schedule_repair_steps"] = (
            runtime.counters.get("draft_schedule_repair_steps", 0) + 1
        )
        for row in range(count):
            if not active[row]:
                continue
            tokens[row].append(token[row, 0].clone())
            scores[row].append(logprob[row, 0].clone())
            lengths[row] += 1
            coordinates[row] += 1
            if values[row] in policy.eos_token_ids:
                reasons[row], active[row] = ("eos", False)
            else:
                parsed = policy.parse_actions(
                    policy.tokenizer.decode(
                        torch.stack(tokens[row]).tolist(), skip_special_tokens=True
                    ).strip()
                )
                if parsed.valid and parsed.actions[-1].name == "stop":
                    reasons[row], active[row] = ("stop", False)
        if not any(active):
            break
    _check_cancel(cancelled)
    return runtime._finish_generation(prefix, tokens, scores, lengths, coordinates, reasons)


def _repair_tokens(
    runtime: ActiveVLNBatchedRuntime,
    prefix: ActiveVLNBatchPrefix,
    trial_tokens: _TensorRows,
    trial_scores: _TensorRows,
    features: _TensorRows,
    trial_reasons: list[str],
    start: int,
    cancelled: Callable[[], bool] | None,
) -> tuple[ActiveVLNGeneration, ...]:
    """Repair an uncommitted suffix while retaining the last validated prefix."""
    policy = runtime.policy
    count = len(trial_tokens)
    batch = runtime.batch_size
    device = runtime.device
    end = max(map(len, trial_tokens))
    limit = policy.max_context if runtime.pooled_kv else runtime.storage.shape[-2]
    first_ids = torch.stack([history[0] for history in prefix.token_history])
    causal = torch.ones(16, 16, device=device, dtype=torch.bool).tril()
    while start < end:
        if cancelled is not None and cancelled():
            raise SessionCancelledError("parallel repair cancelled")
        active = [len(row) > start for row in trial_tokens]
        logits = (
            prefix.logits
            if start == 0
            else policy._lm_head(
                torch.stack([features[row][min(start, len(trial_tokens[row])) - 1] for row in range(count)])
            )
        )
        seen = torch.zeros_like(logits, dtype=torch.bool)
        for row, history in enumerate(prefix.token_history):
            seen[row].scatter_(0, history, True)
            if start:
                seen[row].scatter_(0, torch.stack(trial_tokens[row][:start]), True)
        penalty = policy.repetition_penalty
        effective = torch.where(seen, torch.where(logits < 0, logits * penalty, logits / penalty), logits)
        roots = effective.argmax(-1)
        root_scores = torch.log_softmax(effective, -1).gather(1, roots[:, None])[:, 0]
        if any(active[r] and int(roots[r]) != int(trial_tokens[r][start]) for r in range(count)):
            runtime.counters["parallel_repair_root_fallbacks"] = (
                runtime.counters.get("parallel_repair_root_fallbacks", 0) + 1
            )
            return _repair_serial(
                runtime, prefix, trial_tokens, trial_scores, features, trial_reasons, start, cancelled
            )
        input_ids = torch.zeros(batch, 16, device=device, dtype=torch.long)
        positions = torch.zeros(3, batch, 16, device=device, dtype=torch.long)
        offsets = [0] * batch
        writes = [0] * batch
        for r in range(count):
            if active[r]:
                n = min(16, len(trial_tokens[r]) - start)
                input_ids[r, :n] = torch.stack(trial_tokens[r][start : start + n])
                offsets[r] = prefix.lengths[r] + start
                writes[r] = n
                positions[:, r] = prefix.positions[r] + start + torch.arange(16, device=device)
        buckets = []
        for i in range(start, start + 16):
            if i >= end:
                buckets.append(buckets[-1])
                continue
            latest = max([prefix.lengths[r] + i for r in range(count) if len(trial_tokens[r]) > i] or [0])
            buckets.append(min(limit, max(512, 1 << latest.bit_length())))
        hidden = runtime._serial_draft.forward(
            policy._text.embed_tokens(input_ids), positions, offsets, writes, buckets
        )
        node_logits = runtime._serial_draft.logits(hidden[:count])
        proposed = input_ids[:count]
        chosen, node_scores = speculation.batched_tree_greedy_scores(
            node_logits,
            seen,
            proposed[:, None].expand(-1, 16, -1),
            causal[None].expand(count, -1, -1),
            first_ids,
            penalty,
        )
        choices = chosen.tolist()
        proposed_ids = proposed.tolist()
        accepted = min(16, end - start)
        for r in range(count):
            for j in range(1, min(16, len(trial_tokens[r]) - start)):
                if choices[r][j - 1] != proposed_ids[r][j]:
                    accepted = min(accepted, j)
                    break
        for r in range(count):
            for j in range(min(accepted, len(trial_tokens[r]) - start)):
                features[r][start + j] = hidden[r, j].clone()
                trial_scores[r][start + j] = (
                    root_scores[r].clone() if j == 0 else node_scores[r, j - 1].clone()
                )
        runtime.counters["parallel_repair_blocks"] = runtime.counters.get("parallel_repair_blocks", 0) + 1
        start += accepted
        if accepted < min(16, end - (start - accepted)):
            runtime.counters["parallel_repair_suffix_fallbacks"] = (
                runtime.counters.get("parallel_repair_suffix_fallbacks", 0) + 1
            )
            return _repair_serial(
                runtime, prefix, trial_tokens, trial_scores, features, trial_reasons, start, cancelled
            )
    _check_cancel(cancelled)
    return runtime._finish_generation(
        prefix,
        trial_tokens,
        trial_scores,
        [prefix.lengths[r] + len(trial_tokens[r]) for r in range(count)],
        [prefix.positions[r] + len(trial_tokens[r]) for r in range(count)],
        trial_reasons,
    )


def generate_serial_draft_tokens(
    runtime: ActiveVLNBatchedRuntime,
    prefix: ActiveVLNBatchPrefix,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[ActiveVLNGeneration, ...]:
    """Verify learned linear blocks and retain only accepted per-row KV prefixes."""
    if prefix.last_hidden is None:
        raise ValueError("learned drafting requires target prefix features")
    policy, device = (runtime.policy, runtime.device)
    stop_triggers = runtime._serial_draft.stop_trigger_ids
    count, batch = (len(prefix.prepared.turns), runtime.batch_size)
    lengths, coordinates = (list(prefix.lengths), list(prefix.positions))
    logits, features = (prefix.logits, prefix.last_hidden)
    seen = torch.zeros_like(logits, dtype=torch.bool)
    for row, history in enumerate(prefix.token_history):
        seen[row].scatter_(0, history, True)
    first_ids = torch.stack([history[0] for history in prefix.token_history])
    tokens: list[list[torch.Tensor]] = [[] for _ in range(count)]
    scores: list[list[torch.Tensor]] = [[] for _ in range(count)]
    ids: list[list[int]] = [[] for _ in range(count)]
    verifier_buckets = [[] for _ in range(count)]
    verifier_features = [[] for _ in range(count)]
    predicted_lengths = [None] * count
    limit = policy.max_context if runtime.pooled_kv else runtime.storage.shape[-2]
    reasons, active = (["max_tokens"] * count, [True] * count)

    def check_cancel() -> None:
        _check_cancel(cancelled)

    while any(active):
        check_cancel()
        remaining = min(
            min(policy.max_context, runtime._row_capacities[row]) - lengths[row]
            for row in range(count)
            if active[row]
        )
        if remaining < 1:
            raise ValueError("batched generation exceeds the model context limit")
        query = runtime.draft.block_size if remaining >= runtime.draft.block_size else 1
        penalty = policy.repetition_penalty
        effective = torch.where(seen, torch.where(logits < 0, logits * penalty, logits / penalty), logits)
        roots = effective.argmax(-1)
        root_scores = torch.log_softmax(effective, -1).gather(1, roots[:, None])[:, 0]
        proposed = runtime._propose_tokens(features, roots)[:, :query] if query > 1 else roots[:, None]
        proposed_ids = proposed.tolist()
        terminal_roots = [
            speculation._stop_reason(policy, ids[row] + [proposed_ids[row][0]])
            if active[row]
            else reasons[row]
            for row in range(count)
        ]
        terminal_only = all(terminal_roots[row] is not None for row in range(count) if active[row])
        input_ids = torch.zeros(batch, query, device=device, dtype=torch.long)
        input_ids[:count] = proposed
        positions = torch.zeros(3, batch, query, device=device, dtype=torch.long)
        offsets, writes = ([0] * batch, [0] * batch)
        for row in range(count):
            if active[row]:
                offsets[row], writes[row] = (lengths[row], query)
                positions[:, row] = coordinates[row] + torch.arange(query, device=device)
        causal = torch.ones(query, query, device=device, dtype=torch.bool).tril()
        for row in range(count):
            if not active[row]:
                predicted_lengths[row] = len(ids[row])
                continue
            predicted_lengths[row] = None
            for node in range(query):
                if (
                    terminal_roots[row]
                    if node == 0
                    else "eos"
                    if proposed_ids[row][node] in policy.eos_token_ids
                    else speculation._stop_reason(policy, ids[row] + proposed_ids[row][: node + 1])
                    if proposed_ids[row][node] in stop_triggers
                    else None
                ) is not None or len(ids[row]) + node + 1 >= policy.max_new_tokens:
                    predicted_lengths[row] = len(ids[row]) + node + 1
                    break
        buckets = []
        for row in range(count):
            row_buckets = []
            for node in range(query):
                global_index = len(ids[row]) + node
                possible = [
                    prefix.lengths[other] + global_index
                    for other in range(count)
                    if predicted_lengths[other] is None or predicted_lengths[other] > global_index
                ]
                row_buckets.append(
                    min(limit, max(512, 1 << max(possible).bit_length()))
                    if possible
                    else row_buckets[-1]
                    if row_buckets
                    else 512
                )
            buckets.append(row_buckets)
        if terminal_only:
            buckets = [[row[0]] * query for row in buckets]
        buckets += [buckets[0]] * (batch - count)
        if query > 1:
            hidden = runtime._serial_draft.forward(
                policy._text.embed_tokens(input_ids), positions, offsets, writes, buckets
            )
        else:
            hidden = runtime._forward(
                policy._text.embed_tokens(input_ids), positions, offsets, write_lengths=writes
            )
            actual_bucket = min(limit, max(512, 1 << max(offsets).bit_length()))
            buckets = [[actual_bucket] for _ in range(batch)]
        block_features = hidden[:count].clone()
        if terminal_only:
            check_cancel()
            for row in range(count):
                if not active[row]:
                    continue
                tokens[row].append(proposed[row, 0])
                scores[row].append(root_scores[row])
                ids[row].append(proposed_ids[row][0])
                verifier_buckets[row].append(buckets[row][0])
                verifier_features[row].append(block_features[row, 0])
                lengths[row] += 1
                coordinates[row] += 1
                reasons[row], active[row] = (terminal_roots[row], False)
            runtime.counters["draft_terminal_blocks"] = runtime.counters.get("draft_terminal_blocks", 0) + 1
            break
        node_logits = runtime._serial_draft.logits(hidden[:count])
        chosen, node_scores = speculation.batched_tree_greedy_scores(
            node_logits,
            seen,
            proposed[:, None].expand(-1, query, -1),
            causal[None].expand(count, -1, -1),
            first_ids,
            penalty,
        )
        choices = chosen.tolist()
        runtime.counters["draft_verification_calls"] += 1
        runtime.counters["draft_proposed_tokens"] += sum(active) * (query - 1)
        next_logits, next_features = ([], [])
        for row in range(count):
            if not active[row]:
                next_logits.append(logits[row])
                next_features.append(features[row])
                continue
            accepted = 0
            for node in range(query):
                check_cancel()
                if node and choices[row][node - 1] != proposed_ids[row][node]:
                    break
                accepted += 1
                verifier_buckets[row].append(buckets[row][node])
                verifier_features[row].append(block_features[row, node])
                tokens[row].append(proposed[row, node])
                scores[row].append(root_scores[row] if node == 0 else node_scores[row, node - 1])
                ids[row].append(proposed_ids[row][node])
                reason = (
                    terminal_roots[row]
                    if node == 0
                    else "eos"
                    if ids[row][-1] in policy.eos_token_ids
                    else speculation._stop_reason(policy, ids[row])
                    if ids[row][-1] in stop_triggers
                    else None
                )
                if reason is not None or len(ids[row]) >= policy.max_new_tokens:
                    reasons[row], active[row] = (reason or "max_tokens", False)
                    break
            runtime.counters["draft_accepted_tokens"] += accepted - 1
            lengths[row] += accepted
            coordinates[row] += accepted
            seen[row].scatter_(0, proposed[row, :accepted], True)
            next_logits.append(node_logits[row, accepted - 1])
            next_features.append(hidden[row, accepted - 1])
        logits, features = (torch.stack(next_logits), torch.stack(next_features))
    invalid = _first_invalid_context(prefix.lengths, list(map(len, tokens)), verifier_buckets, limit)
    if invalid is not None:
        runtime.counters["draft_schedule_repaired_batches"] = (
            runtime.counters.get("draft_schedule_repaired_batches", 0) + 1
        )
        runtime.counters["draft_schedule_repair_starts_sum"] = (
            runtime.counters.get("draft_schedule_repair_starts_sum", 0) + invalid
        )
        return _repair_tokens(runtime, prefix, tokens, scores, verifier_features, reasons, invalid, cancelled)
    check_cancel()
    return runtime._finish_generation(prefix, tokens, scores, lengths, coordinates, reasons)
