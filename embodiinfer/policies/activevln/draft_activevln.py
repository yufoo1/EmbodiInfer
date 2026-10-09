"""Learned feature-conditioned multi-token proposals for ActiveVLN.

The restricted output vocabulary only limits proposals. Target verification must
always use its complete vocabulary and may reject every proposed suffix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ActiveVLNDraft(nn.Module):
    """Predict a suffix from target features and the already known first token."""

    def __init__(
        self, hidden_size: int, token_ids: list[int], *, width: int = 512, block_size: int = 16
    ) -> None:
        super().__init__()
        if hidden_size < 1 or width < 1 or not 2 <= block_size <= 32:
            raise ValueError("invalid draft dimensions")
        if not token_ids or token_ids != sorted(set(token_ids)) or token_ids[0] < 0:
            raise ValueError("draft vocabulary must be sorted, unique nonnegative token IDs")
        self.hidden_size, self.width, self.block_size = hidden_size, width, block_size
        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long))
        lookup = torch.full((token_ids[-1] + 1,), len(token_ids), dtype=torch.long)
        lookup[self.token_ids] = torch.arange(len(token_ids))
        self.register_buffer("lookup", lookup)
        self.norm = nn.LayerNorm(hidden_size)
        self.feature = nn.Linear(hidden_size, width, bias=False)
        self.root = nn.Embedding(len(token_ids) + 1, width)
        self.residual = nn.Linear(width, width)
        self.head = nn.Linear(width, (block_size - 1) * len(token_ids))

    def forward(self, hidden: torch.Tensor, roots: torch.Tensor) -> torch.Tensor:
        """Return suffix logits [batch, block_size-1, draft_vocab]; roots stay fixed."""
        valid = (roots >= 0) & (roots < self.lookup.numel())
        indices = self.lookup[roots.clamp(0, self.lookup.numel() - 1)]
        indices = torch.where(valid, indices, self.token_ids.numel())
        features = F.silu(self.feature(self.norm(hidden)) + self.root(indices))
        features = features + F.silu(self.residual(features))
        return self.head(features).view(hidden.shape[0], self.block_size - 1, self.token_ids.numel())

    def propose(self, hidden: torch.Tensor, roots: torch.Tensor) -> torch.Tensor:
        """Return a block beginning with the full-vocabulary target's chosen root."""
        suffix = self.token_ids[self(hidden, roots).argmax(-1)]
        return torch.cat((roots[:, None], suffix), dim=1)

    def checkpoint(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Create a weights-only-loadable artifact with training provenance."""
        return {
            "schema": "activevln.feature_draft.v1",
            "config": {
                "hidden_size": self.hidden_size,
                "width": self.width,
                "block_size": self.block_size,
                "token_ids": self.token_ids.cpu().tolist(),
            },
            "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
            "metadata": metadata,
        }

    @classmethod
    def load(cls, path: str | Path) -> tuple[ActiveVLNDraft, dict[str, Any]]:
        """Load only tensors/primitives and reject an unrecognized draft schema."""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("schema") != "activevln.feature_draft.v1":
            raise ValueError("unsupported ActiveVLN draft checkpoint")
        model = cls(**saved["config"])
        model.load_state_dict(saved["state_dict"], strict=True)
        return model.eval(), saved["metadata"]
