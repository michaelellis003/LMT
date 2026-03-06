r"""Sliding Window Attention.

Each token attends only to the previous ``w`` tokens (including
itself), where ``w`` is the window size. This reduces per-layer
complexity from :math:`O(n^2)` to :math:`O(n \cdot w)`.

The effective receptive field grows linearly with depth:
after ``L`` layers, a token can attend to positions up to
``L * w`` tokens back. Used in Mistral and Mixtral.

When ``window_size >= seq_len``, this is equivalent to standard
causal attention.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from lmt.models.config import ModelConfig


class SlidingWindowAttention(nn.Module):
    r"""Multi-head attention with a sliding window causal mask.

    Args:
        model_config: Model configuration with embed_dim, num_heads,
            context_length, window_size, and qkv_bias.
    """

    causal_mask: Tensor

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize SlidingWindowAttention.

        Args:
            model_config: Model configuration. window_size defaults
                to context_length if not set (full causal attention).
        """
        super().__init__()
        embed_dim = model_config.embed_dim
        num_heads = model_config.num_heads
        ctx = model_config.context_length
        self.window_size = model_config.window_size or ctx

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        bias = model_config.qkv_bias
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Build sliding window causal mask
        mask = torch.full((ctx, ctx), float('-inf'))
        for i in range(ctx):
            # Attend to positions max(0, i - w + 1) through i
            start = max(0, i - self.window_size + 1)
            mask[i, start : i + 1] = 0.0
        self.register_buffer('causal_mask', mask)

    def forward(self, x: Tensor) -> Tensor:
        """Apply sliding window attention.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output with same shape.
        """
        b, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, seq_len, self.num_heads, self.head_dim)
        k = k.view(b, seq_len, self.num_heads, self.head_dim)
        v = v.view(b, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:seq_len, :seq_len]
        attn = attn + mask

        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)

        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, seq_len, -1)
        return self.out_proj(out)
