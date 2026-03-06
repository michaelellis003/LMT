r"""RMSNorm (Root Mean Square Layer Normalization).

Implements the normalization described in:
    Zhang & Sennrich, 2019 -- "Root Mean Square Layer Normalization"

.. math::

    \text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\bar{x^2} + \epsilon}}

Unlike LayerNorm, RMSNorm skips the mean-centering step and only
normalizes by the RMS statistic. This makes it ~10-15% faster while
performing comparably in practice. The intuition: the learned gain
parameter (gamma) can compensate for any mean shift, so explicitly
subtracting the mean is redundant.

Used in LLaMA, Mistral, and most modern transformer architectures.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Args:
        d_model: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            d_model: Feature dimension to normalize over.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_model]``.

        Returns:
            Normalized tensor with the same shape as input.
        """
        input_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).to(input_dtype)
