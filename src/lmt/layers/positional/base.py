"""Base protocol for positional encoding layers."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class BasePositionalEncoding(Protocol):
    """Protocol for positional encoding components.

    Positional encoding implementations may operate in different ways
    (additive embeddings, rotary transformations, etc.), but they all
    accept input tensors of shape ``[batch, seq_len, d_model]`` and
    return tensors of the same shape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_model]``.

        Returns:
            Positionally-encoded tensor of shape
            ``[batch, seq_len, d_model]``.
        """
        ...
