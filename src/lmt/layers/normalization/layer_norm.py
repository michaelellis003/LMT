"""LayerNorm wrapper for consistency with the normalization interface."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    r"""Layer Normalization wrapper.

    Wraps ``torch.nn.LayerNorm`` to provide a consistent interface within
    the LMT normalization component registry.

    .. math::

        \text{LayerNorm}(x) = \gamma \cdot
        \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

    where :math:`\mu` and :math:`\sigma^2` are the mean and variance
    computed over the last dimension.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """Initialize LayerNorm.

        Args:
            d_model: The dimension to normalize over.
            eps: A small value added for numerical stability.
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    @property
    def normalized_shape(self) -> tuple[int, ...]:
        """Return the normalized shape for compatibility."""
        return self.norm.normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_model]``.

        Returns:
            Normalized tensor of shape ``[batch, seq_len, d_model]``.
        """
        return self.norm(x)
