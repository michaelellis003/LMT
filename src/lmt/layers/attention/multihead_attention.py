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
"""

import math

import torch
import torch.nn as nn

from lmt.models.config import ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer."""

    # Correctly type self.mask as a Tensor
    causal_mask: torch.Tensor

    def __init__(self, model_config: ModelConfig):
        """Initialize the Multi-Head Attention layer.

        Uses ``num_heads`` parallel attention heads with causal masks.
        A fused QKV projection splits into Q, K, V which are reshaped
        into per-head tensors before scaled dot-product attention.

        Args:
            model_config (ModelConfig): Configuration object containing model
                parameters.

        """
        super().__init__()
        assert model_config.embed_dim % model_config.num_heads == 0, (
            'embed_dim must be divisible by num_heads'
        )
        self.embed_dim = model_config.embed_dim
        self.num_heads = model_config.num_heads
        self.head_dim = model_config.embed_dim // model_config.num_heads

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
            x (torch.Tensor): Token embedding vectors with shape (batch_size,
                seq_length, embed_dim). Assumed to have position vectors
                added previously or no position vectors will be added.

        Returns:
            torch.Tensor: Context vector, a reweighting of the input token
                embedding vectors according to the multi-head scaled
                dot-product attention mechanism with a causal mask and the
                same dimension as the input (batch_size, seq_length,
                embed_dim).

        """
        b, seq_length, _ = x.shape

        # With fused qkv we can compute the query, keys and values in a
        # single matrix multiplication instead of three separate ones
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * embed_dim]
        queries, keys, values = qkv.chunk(3, dim=-1)

        keys = keys.view(b, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(b, seq_length, self.num_heads, self.head_dim)
        values = values.view(b, seq_length, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores * self.scale

        # Prevent attention to future tokens in autoregressive modeling
        causal_mask = self.causal_mask[:seq_length, :seq_length]
        attn_scores = attn_scores + causal_mask

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
