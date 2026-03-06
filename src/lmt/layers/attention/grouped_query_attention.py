r"""Grouped Query Attention (GQA).

Implements the attention mechanism from:
    Ainslie et al., 2023 -- "GQA: Training Generalized Multi-Query
    Transformer Models from Multi-Head Checkpoints"

.. math::

    \text{GQA}(Q, K, V) = \text{softmax}\!\left(
        \frac{Q K^\top}{\sqrt{d_k}}
    \right) V

K and V have ``num_kv_heads`` (fewer than ``num_heads``), and each
KV head is shared across ``num_heads // num_kv_heads`` query heads.

Special cases:
- ``num_kv_heads == num_heads``: standard Multi-Head Attention
- ``num_kv_heads == 1``: Multi-Query Attention (MQA)

The key insight: K,V projections are the memory bottleneck in
inference (KV cache). By sharing them across query groups, we
drastically reduce cache size with minimal quality loss.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from lmt.models.config import ModelConfig


class GroupedQueryAttention(nn.Module):
    r"""Grouped Query Attention.

    Args:
        model_config: Model configuration with embed_dim, num_heads,
            num_kv_heads, context_length, and qkv_bias.
    """

    causal_mask: Tensor

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize Grouped Query Attention.

        Args:
            model_config: Model configuration. If num_kv_heads is
                None, defaults to num_heads (standard MHA).
        """
        super().__init__()
        embed_dim = model_config.embed_dim
        num_heads = model_config.num_heads
        num_kv_heads = model_config.num_kv_heads
        if num_kv_heads is None:
            num_kv_heads = num_heads

        assert embed_dim % num_heads == 0, (
            'embed_dim must be divisible by num_heads'
        )
        assert num_heads % num_kv_heads == 0, (
            'num_heads must be divisible by num_kv_heads'
        )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_heads // num_kv_heads

        bias = model_config.qkv_bias

        self.q_proj = nn.Linear(
            embed_dim, num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            embed_dim, num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            embed_dim, num_kv_heads * self.head_dim, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.register_buffer(
            'causal_mask',
            torch.full(
                (
                    model_config.context_length,
                    model_config.context_length,
                ),
                float('-inf'),
            ).triu(diagonal=1),
        )

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply grouped query attention.

        Args:
            x: Input of shape ``[batch, seq_len, embed_dim]``.

        Returns:
            Output tensor with same shape as input.
        """
        b, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: [b, seq, heads, head_dim] -> [b, heads, seq, head_dim]
        q = q.view(b, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        k = k.view(b, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = v.view(b, seq_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)

        # Expand KV heads to match Q heads by repeating
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:seq_len, :seq_len]
        attn = attn + mask

        # Upcast to FP32 for softmax stability
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)

        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, seq_len, -1)
        return self.out_proj(out)
