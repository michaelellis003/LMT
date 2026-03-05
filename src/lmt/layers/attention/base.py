"""Base protocol for attention mechanisms."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class BaseAttention(Protocol):
    """Protocol for attention mechanisms.

    All attention implementations must accept input tensors of shape
    ``[batch, seq_len, d_model]`` and return tensors of the same shape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention over the input.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_model]``.

        Returns:
            Output tensor of shape ``[batch, seq_len, d_model]``.
        """
        ...
