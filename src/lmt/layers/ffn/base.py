"""Base protocol for feed-forward networks."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class BaseFeedForward(Protocol):
    """Protocol for feed-forward network components.

    All FFN implementations must accept input tensors of shape
    ``[batch, seq_len, d_model]`` and return tensors of the same shape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_model]``.

        Returns:
            Output tensor of shape ``[batch, seq_len, d_model]``.
        """
        ...
