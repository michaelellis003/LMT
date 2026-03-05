"""Base protocol for normalization layers."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class BaseNorm(Protocol):
    """Protocol for normalization layers.

    All normalization implementations must accept input tensors of shape
    ``[batch, seq_len, d_model]`` and return tensors of the same shape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_model]``.

        Returns:
            Normalized tensor of shape ``[batch, seq_len, d_model]``.
        """
        ...
