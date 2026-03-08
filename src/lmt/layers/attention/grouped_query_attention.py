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

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from lmt.layers.attention.kv_cache import KVCache
from lmt.models.config import ModelConfig

if TYPE_CHECKING:
    from lmt.layers.positional import RoPE


class GroupedQueryAttention(nn.Module):
    r"""Grouped Query Attention.

    Supports optional RoPE integration and sliding window masking,
    making it composable enough to serve as the attention layer for
    LLaMA (GQA + RoPE), Mixtral (GQA + RoPE + sliding window), and
    standard GPT (no RoPE, no window).

    Args:
        model_config: Model configuration with embed_dim, num_heads,
            num_kv_heads, context_length, and qkv_bias.
        rope: Optional RoPE instance for rotary positional encoding.
            When provided, RoPE is applied to Q and K inside the
            attention computation.
        window_size: Optional sliding window size. When set, each
            token only attends to the previous ``window_size`` tokens.
    """

    causal_mask: Tensor

    def __init__(
        self,
        model_config: ModelConfig,
        rope: RoPE | None = None,
        window_size: int | None = None,
    ) -> None:
        """Initialize Grouped Query Attention.

        Args:
            model_config: Model configuration. If num_kv_heads is
                None, defaults to num_heads (standard MHA).
            rope: Optional RoPE instance for Q/K rotation.
            window_size: Optional sliding window size.
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
        self.rope = rope

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

        # QK-Norm: normalize Q and K before dot product (Qwen3, Gemma 3)
        self.qk_norm = getattr(model_config, 'qk_norm', False)
        if self.qk_norm:
            from lmt.layers.normalization.rms_norm import RMSNorm

            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Build mask: causal with optional sliding window
        ctx = model_config.context_length
        if window_size is not None:
            mask = torch.full((ctx, ctx), float('-inf'))
            for i in range(ctx):
                start = max(0, i - window_size + 1)
                mask[i, start : i + 1] = 0.0
        else:
            mask = torch.full((ctx, ctx), float('-inf')).triu(diagonal=1)
        self.register_buffer('causal_mask', mask)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Optional KV cache for autoregressive generation.
        # Set to a KVCache instance to enable caching.
        self.kv_cache: KVCache | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply grouped query attention.

        When ``kv_cache`` is set, new K/V are appended to the cache
        and the full cached K/V are used for attention. This enables
        efficient autoregressive generation.

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
        k = k.view(b, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(b, seq_len, self.num_kv_heads, self.head_dim)

        # QK-Norm: normalize per-head before RoPE and dot product
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE to Q and K if configured
        if self.rope is not None:
            # RoPE expects [batch, seq, dim], so reshape per-head
            q = q.permute(0, 2, 1, 3).reshape(
                b * self.num_heads, seq_len, self.head_dim
            )
            k = k.permute(0, 2, 1, 3).reshape(
                b * self.num_kv_heads, seq_len, self.head_dim
            )
            q, k = self.rope.apply_rotary_emb(q, k)
            q = q.view(b, self.num_heads, seq_len, self.head_dim)
            k = k.view(b, self.num_kv_heads, seq_len, self.head_dim)
            v = v.permute(0, 2, 1, 3)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # Update KV cache if active
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(k, v)

        # Expand KV heads to match Q heads by repeating
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention with causal mask
        kv_len = k.shape[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Mask: query positions are the last seq_len of kv_len
        q_start = kv_len - seq_len
        mask = self.causal_mask[q_start:kv_len, :kv_len]
        attn = attn + mask

        # Upcast to FP32 for softmax stability
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)

        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, seq_len, -1)
        return self.out_proj(out)
