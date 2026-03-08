r"""ReLU² GLU (Squared ReLU Gated Linear Unit) feed-forward network.

Replaces SiLU gating (SwiGLU) with squared ReLU:

.. math::

    \text{ReluSquaredGLU}(x) = W_2 \bigl(\text{ReLU}(x W_g)^2 \odot x W_1\bigr)

Squared ReLU creates **sparser activations** than SiLU:
- ReLU zeros all negative values (~50% sparsity for Gaussian inputs)
- Squaring further suppresses small positive values while amplifying
  large ones, creating a sharper gating signal

Used in Karpathy's modern GPT recipe (autoresearch, 2025) and
So et al. (2021) "Primer: Searching for Efficient Transformers."

Same parameter count as SwiGLU -- drop-in replacement with different
activation characteristics.
"""

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as f


class ReluSquaredGLU(nn.Module):
    r"""Squared ReLU GLU feed-forward network.

    Drop-in replacement for SwiGLU with sparser activations.
    Uses :math:`\text{ReLU}(x)^2` instead of :math:`\text{SiLU}(x)`.

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
        """Initialize ReluSquaredGLU FFN.

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
        """Apply ReluSquaredGLU to input tensor.

        Args:
            x: Input of shape ``[batch, seq_len, d_model]``.

        Returns:
            Output tensor with same shape as input.
        """
        return self.w2(f.relu(self.wg(x)) ** 2 * self.w1(x))
