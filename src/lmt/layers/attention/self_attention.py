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

"""Implementation of Self-Attention layer."""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Class to implement scaled dot-product self-attention mechanism.

    A version of this implementation was originally proposed in "Neural
    Machine Translation by Jointly Learning to Align and Translate
    Bahdanau et al., 2016 and further refined in "Attention Is All You Need"
    by Vaswani et al., 2017. This class is based on the ideas proposed in
    Vaswani et al., 2017.
    """

    def __init__(self, embed_dim: int, qkv_bias: bool = False):
        """Initialize the self attention module.

        Args:
            embed_dim (int): Dimension of token embedding vectors
            qkv_bias (bool): Whether to use bias in query, key, and value
                projections.
        """
        super().__init__()
        self.W_query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for scaled dot-product self-attention mechanism.

        Args:
            x (torch.Tensor): Token embedding vectors with shape (batch_size,
                seq_length, embed_dim). Assumed to have position vectors
                added previously or no position vectors will be added.

        Returns:
            torch.Tensor: Context vector, a reweighting of the input token
                embedding vectors according to the scaled dot-product
                attention mechanism with the same dimension as the input
                (batch_size, seq_length, embed_dim).
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / keys.shape[-1] ** 0.5
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(
            queries.dtype
        )

        return attn_weights @ values
