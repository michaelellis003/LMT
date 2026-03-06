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

"""Implementation of Causal Attention layer."""

import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    """Class for scaled dot-product self-attention with a causal mask."""

    # Correctly type self.mask as a Tensor
    mask: torch.Tensor

    def __init__(
        self, embed_dim: int, context_length: int, qkv_bias: bool = False
    ):
        """Initialize scaled dot-product self-attention with a causal mask.

        Same scaled dot-product self-attention but a "mask" is included
        so that context vector, yn, is computed using only token embedding
        vectors, x0, x1, ..., xn, so that no information is "leaked"
        when predicting the next token in a sequence.

        Args:
            embed_dim (int): Dimension of token embedding vectors.
            context_length (int): Max length of input sequence.
            qkv_bias (bool): Whether to use bias in query, key, and value
                projections.
        """
        super().__init__()

        self.W_query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.register_buffer(
            'mask',
            torch.triu(
                torch.ones((context_length, context_length), dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal attention.

        Args:
            x (torch.Tensor): Token embedding vectors with shape (batch_size,
                seq_length, embed_dim). Assumed to have position vectors
                added previously or no position vectors will be added.

        Returns:
            torch.Tensor: Context vector, a reweighting of the input token
                embedding vectors according to the scaled dot-product
                attention mechanism with a causal mask and the same dimension
                as the input (batch_size, seq_length, embed_dim).
        """
        _, seq_length, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / keys.shape[-1] ** 0.5
        attn_scores.masked_fill_(
            self.mask.bool()[:seq_length, :seq_length], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(
            queries.dtype
        )

        return attn_weights @ values
