r"""SwiGLU feed-forward network.

Implements the gated linear unit with Swish activation from:
    Shazeer, 2020 -- "GLU Variants Improve Transformer"

.. math::

    \text{SwiGLU}(x) = W_2 \bigl(\text{SiLU}(x W_g) \odot x W_1\bigr)

where :math:`\text{SiLU}(x) = x \cdot \sigma(x)` (Swish activation).

The gating mechanism lets the network learn *which* features to
pass through, rather than applying a fixed nonlinearity. Three
projection matrices (W1, Wg, W2) replace the standard two-matrix
FFN, but the hidden dimension is typically reduced to 2/3 of what
a standard FFN would use, keeping total parameter count similar.

All linear layers are bias-free, following modern practice
(LLaMA, Mistral, etc.).
"""

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as f


class SwiGLU(nn.Module):
    r"""SwiGLU feed-forward network.

    Args:
        d_model: Input and output feature dimension.
        hidden_dim: Intermediate dimension. If None, defaults
            to ``int(2/3 * 4 * d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
    ) -> None:
        """Initialize SwiGLU FFN.

        Args:
            d_model: Input and output feature dimension.
            hidden_dim: Intermediate dimension. Defaults to
                ``int(2/3 * 4 * d_model)`` if not specified.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 / 3 * 4 * d_model)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.wg = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU to input tensor.

        Args:
            x: Input of shape ``[batch, seq_len, d_model]``.

        Returns:
            Output tensor with same shape as input.
        """
        return self.w2(f.silu(self.wg(x)) * self.w1(x))
