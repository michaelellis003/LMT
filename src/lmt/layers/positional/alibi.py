r"""ALiBi -- Attention with Linear Biases.

Implements the position encoding method from:
    Press et al., 2022 -- "Train Short, Test Long: Attention with
    Linear Biases Enables Input Length Extrapolation" (ICLR 2022)

Instead of adding positional embeddings to token representations,
ALiBi adds a **static bias** to attention scores that penalizes
distant query-key pairs:

.. math::

    \text{Attention}(Q, K, V) = \text{softmax}\!\left(
        \frac{Q K^\top}{\sqrt{d_k}} + B
    \right) V

where :math:`B_{h,i,j} = -m_h \cdot |i - j|` and :math:`m_h` is a
head-specific slope from a geometric sequence.

The slopes are set as:

.. math::

    m_h = 2^{-\frac{8}{n} \cdot h}, \quad h = 1, 2, \ldots, n

where :math:`n` is the number of attention heads. This gives slopes
in :math:`(0, 1)` with density increasing near 0.

Key properties:
- **No learnable parameters** -- the biases are entirely static.
- **Recency bias** -- closer tokens get higher attention weight.
- **Length extrapolation** -- models trained on short sequences
  generalize to longer ones because the bias is a simple linear
  function of distance, not a learned embedding.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ALiBi(nn.Module):
    r"""Attention with Linear Biases for position encoding.

    Precomputes a bias matrix of shape ``[num_heads, max_seq_len,
    max_seq_len]`` where ``bias[h, i, j] = -slope_h * |i - j|``.

    Use :meth:`get_bias` to extract the bias for a given sequence
    length, then add it to attention scores before softmax.

    Args:
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length to precompute.

    Example::

        alibi = ALiBi(num_heads=8, max_seq_len=2048)

        # Inside attention:
        attn_scores = (q @ k.T) / sqrt(d_k)
        attn_scores = attn_scores + alibi.get_bias(seq_len)
        attn_weights = softmax(attn_scores)
    """

    slopes: Tensor
    bias: Tensor

    def __init__(self, num_heads: int, max_seq_len: int) -> None:
        """Initialize ALiBi.

        Args:
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()
        self.num_heads = num_heads

        # Compute slopes: geometric sequence 2^(-8/n * h)
        slopes = self._compute_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # Precompute distance matrix and bias
        # distances[i, j] = |i - j|
        positions = torch.arange(max_seq_len)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # bias[h, i, j] = -slope_h * |i - j|
        bias = -slopes.unsqueeze(1).unsqueeze(2) * distances.unsqueeze(0)
        self.register_buffer('bias', bias)

    @staticmethod
    def _compute_slopes(num_heads: int) -> Tensor:
        """Compute the head-specific slope values.

        For ``n`` heads, slopes follow ``2^(-8/n * h)`` for
        ``h = 1, 2, ..., n``.

        Args:
            num_heads: Number of attention heads.

        Returns:
            Tensor of slopes with shape ``[num_heads]``.
        """
        ratio = 8.0 / num_heads
        slopes = torch.tensor(
            [2.0 ** (-ratio * (h + 1)) for h in range(num_heads)]
        )
        return slopes

    def get_bias(self, seq_len: int) -> Tensor:
        """Extract bias for a given sequence length.

        Args:
            seq_len: Current sequence length (<= max_seq_len).

        Returns:
            Bias tensor ``[num_heads, seq_len, seq_len]``.
        """
        return self.bias[:, :seq_len, :seq_len]
