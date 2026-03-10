# Copyright 2025 Michael Ellis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-Head Attention Layer.

Implements scaled dot-product multi-head attention from:
    Vaswani et al., 2017 -- "Attention Is All You Need"
    https://arxiv.org/abs/1706.03762

Supports optional positional encodings:
- **RoPE**: Rotary embeddings applied to Q and K before dot product.
- **ALiBi**: Linear bias added to attention scores.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from lmt.models.config import ModelConfig

if TYPE_CHECKING:
    from lmt.layers.positional import ALiBi, RoPE


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer.

    Args:
        model_config: Model configuration.
        rope: Optional RoPE instance for rotary positional encoding.
        alibi: Optional ALiBi instance for linear bias encoding.
    """

    # Correctly type self.mask as a Tensor
    causal_mask: torch.Tensor

    def __init__(
        self,
        model_config: ModelConfig,
        rope: RoPE | None = None,
        alibi: ALiBi | None = None,
    ) -> None:
        """Initialize the Multi-Head Attention layer.

        Uses ``num_heads`` parallel attention heads with causal masks.
        A fused QKV projection splits into Q, K, V which are reshaped
        into per-head tensors before scaled dot-product attention.

        Args:
            model_config: Configuration object containing model parameters.
            rope: Optional RoPE for Q/K rotation.
            alibi: Optional ALiBi for attention score bias.

        """
        super().__init__()
        assert model_config.embed_dim % model_config.num_heads == 0, (
            'embed_dim must be divisible by num_heads'
        )
        self.embed_dim = model_config.embed_dim
        self.num_heads = model_config.num_heads
        self.head_dim = model_config.embed_dim // model_config.num_heads
        self.rope = rope
        self.alibi = alibi

        # Using fused qkv
        # This reduces memory overhead
        self.qkv_proj = nn.Linear(
            model_config.embed_dim,
            3 * model_config.embed_dim,
            bias=model_config.qkv_bias,
        )

        self.out_proj = nn.Linear(
            model_config.embed_dim, model_config.embed_dim, bias=False
        )

        self.register_buffer(
            'causal_mask',
            torch.full(
                (model_config.context_length, model_config.context_length),
                float('-inf'),
            ).triu(diagonal=1),
        )

        # Pre-compute scaling factor for efficiency
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Attention weights stored for analysis (not saved in state_dict).
        # Shape: [batch, num_heads, seq_len, seq_len] after forward.
        self._attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multi-Head Attention layer.

        Args:
            x: Token embeddings ``[batch, seq_len, embed_dim]``.

        Returns:
            Context vectors ``[batch, seq_len, embed_dim]``.
        """
        b, seq_length, _ = x.shape

        # Fused QKV projection
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * embed_dim]
        queries, keys, values = qkv.chunk(3, dim=-1)

        queries = queries.view(b, seq_length, self.num_heads, self.head_dim)
        keys = keys.view(b, seq_length, self.num_heads, self.head_dim)
        values = values.view(b, seq_length, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K if configured
        if self.rope is not None:
            q = queries.permute(0, 2, 1, 3).reshape(
                b * self.num_heads, seq_length, self.head_dim
            )
            k = keys.permute(0, 2, 1, 3).reshape(
                b * self.num_heads, seq_length, self.head_dim
            )
            q, k = self.rope.apply_rotary_emb(q, k)
            queries = q.view(b, self.num_heads, seq_length, self.head_dim)
            keys = k.view(b, self.num_heads, seq_length, self.head_dim)
            values = values.permute(0, 2, 1, 3)
        else:
            queries = queries.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores * self.scale

        # Causal mask
        causal_mask = self.causal_mask[:seq_length, :seq_length]
        attn_scores = attn_scores + causal_mask

        # ALiBi: add distance-based bias to attention scores
        if self.alibi is not None:
            attn_scores = attn_scores + self.alibi.get_bias(seq_length).to(
                attn_scores.device
            )

        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(
            queries.dtype
        )
        self._attn_weights = attn_weights.detach()

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, seq_length, self.embed_dim
        )
        context_vec = self.out_proj(context_vec)

        return context_vec
