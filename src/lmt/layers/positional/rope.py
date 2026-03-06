r"""Rotary Positional Encoding (RoPE).

Implements the positional encoding from:
    Su et al., 2021 -- "RoFormer: Enhanced Transformer with Rotary
    Position Embedding"

.. math::

    \theta_i = \text{base}^{-2i/d}

    R(\theta, m) \cdot x = \begin{bmatrix}
        x_1 \cos(m\theta) - x_2 \sin(m\theta) \\
        x_2 \cos(m\theta) + x_1 \sin(m\theta)
    \end{bmatrix}

The key property that makes this work for relative position:

.. math::

    \langle R(m)q,\; R(n)k \rangle = \langle R(m-n)q,\; k \rangle

So the dot product between rotated Q and K depends only on their
*relative* distance, not absolute positions. Unlike additive
positional embeddings, RoPE preserves this through the attention
dot product naturally.

RoPE has no learnable parameters -- it's purely geometric.
Precomputed cos/sin caches are stored as buffers.
"""

import torch
import torch.nn as nn
from torch import Tensor


class RoPE(nn.Module):
    r"""Rotary Positional Encoding.

    Applies rotation to pairs of dimensions based on position.
    No learnable parameters -- position is encoded geometrically.

    Args:
        d_model: Feature dimension (must be even).
        base: Base for frequency computation.
        max_seq_len: Maximum sequence length to precompute.
    """

    def __init__(
        self,
        d_model: int,
        base: float = 10000.0,
        max_seq_len: int = 4096,
    ) -> None:
        """Initialize RoPE.

        Args:
            d_model: Feature dimension (must be even).
            base: Base for frequency computation.
            max_seq_len: Max sequence length to cache.
        """
        super().__init__()
        assert d_model % 2 == 0, 'd_model must be even for RoPE'

        half_d = d_model // 2
        # theta_i = base^(-2i/d) for i in [0, d/2)
        freqs = 1.0 / (
            base ** (torch.arange(0, half_d).float() / half_d)
        )
        # positions: [0, 1, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len).float()
        # angles: [max_seq_len, half_d]
        angles = torch.outer(positions, freqs)

        self.register_buffer('cos_cache', angles.cos())
        self.register_buffer('sin_cache', angles.sin())
        self.cos_cache: Tensor
        self.sin_cache: Tensor

    def _rotate(self, x: Tensor, seq_len: int) -> Tensor:
        """Apply rotary embedding to a single tensor.

        Args:
            x: Tensor of shape ``[batch, seq_len, d_model]``.
            seq_len: Actual sequence length to use.

        Returns:
            Rotated tensor with same shape.
        """
        cos = self.cos_cache[:seq_len].unsqueeze(0)  # [1, seq, d/2]
        sin = self.sin_cache[:seq_len].unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat(
            [x1 * cos - x2 * sin, x2 * cos + x1 * sin],
            dim=-1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply RoPE to input tensor (registry-compatible interface).

        Args:
            x: Input of shape ``[batch, seq_len, d_model]``.

        Returns:
            Rotated tensor with same shape.
        """
        return self._rotate(x, x.shape[1])

    def apply_rotary_emb(
        self, q: Tensor, k: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary embeddings to query and key tensors.

        This is the primary interface for use inside attention.

        Args:
            q: Query tensor ``[batch, seq_len, d_model]``.
            k: Key tensor ``[batch, seq_len, d_model]``.

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes.
        """
        seq_len = q.shape[1]
        return self._rotate(q, seq_len), self._rotate(k, seq_len)
