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
r"""Educational Flash Attention implementation.

This module implements the Flash Attention algorithm from Dao et al. (2022)
as a pure Python/PyTorch educational implementation. It demonstrates the
tiling strategy and online softmax that make Flash Attention IO-efficient.

**Why standard attention is memory-bound:**

Standard attention computes :math:`\text{softmax}(QK^T / \sqrt{d}) \cdot V`
by materializing the full :math:`N \times N` attention score matrix. For
sequence length N=4096 and 32 heads, that's 32 * 4096^2 * 4 bytes = 2 GB
of intermediate memory -- just for one layer.

**How Flash Attention fixes this:**

Instead of computing the full score matrix, Flash Attention processes Q
and K/V in small tiles (blocks). The key challenge is that softmax needs
the global maximum for numerical stability. Flash Attention solves this
with **online softmax** (Milakov & Gimelshein, 2018): it maintains a
running max and sum, rescaling accumulated outputs when a new block
produces larger scores.

The algorithm (simplified):

    For each Q block (rows i):
        Initialize: output O_i = 0, running max m_i = -inf, sum l_i = 0
        For each K,V block (columns j):
            1. Compute local scores: S_ij = Q_i @ K_j^T / sqrt(d)
            2. Apply causal mask if needed
            3. Find local max: m_ij = max(S_ij)
            4. Update global max: m_new = max(m_i, m_ij)
            5. Rescale previous output: O_i *= exp(m_i - m_new) * l_i
            6. Add new contribution: O_i += exp(S_ij - m_new) @ V_j
            7. Update sum: l_i = exp(m_i - m_new)*l_i + sum(exp(S_ij - m_new))
            8. m_i = m_new
        Normalize: O_i /= l_i

This never materializes the full N x N matrix -- only block_size x block_size
tiles exist at any time.

**Important:** This is NOT a fused CUDA kernel. The actual performance
benefit of Flash Attention comes from keeping tiles in fast SRAM (on-chip
memory) and minimizing HBM (off-chip memory) reads/writes. This pure
Python version demonstrates algorithmic correctness, not IO efficiency.

Reference: Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness." https://arxiv.org/abs/2205.14135
"""

import math

import torch
import torch.nn as nn

from lmt.models.config import ModelConfig


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    causal: bool = True,
) -> torch.Tensor:
    r"""Compute attention using the Flash Attention tiling algorithm.

    This is a pure Python implementation of Algorithm 2 from Dao et al.
    (2022). It produces identical results to standard attention but
    processes Q and K/V in tiles, demonstrating how the online softmax
    trick avoids materializing the full :math:`N \times N` score matrix.

    Args:
        q: Query tensor ``[B, H, L, D]``.
        k: Key tensor ``[B, H, L, D]``.
        v: Value tensor ``[B, H, L, D]``.
        block_size: Tile size for blocking Q and K/V. Smaller means less
            peak memory but more iterations. Must be > 0.
        causal: Whether to apply causal (autoregressive) masking.

    Returns:
        Output tensor ``[B, H, L, D]``, identical to standard
        ``softmax(QK^T/sqrt(d)) @ V``.
    """
    b, h, seq_len, d = q.shape
    scale = 1.0 / math.sqrt(d)
    num_q_blocks = math.ceil(seq_len / block_size)
    num_kv_blocks = math.ceil(seq_len / block_size)

    # Process each Q block independently to avoid in-place ops
    # that would break autograd. We accumulate per-block results
    # and cat them at the end.
    q_block_outputs = []

    for i in range(num_q_blocks):
        q_start = i * block_size
        q_end = min(q_start + block_size, seq_len)
        q_block = q[:, :, q_start:q_end]  # [B, H, Br, D]
        br = q_end - q_start

        # Per-Q-block accumulators for online softmax
        acc_output = torch.zeros(b, h, br, d, device=q.device, dtype=q.dtype)
        acc_max = torch.full(
            (b, h, br, 1), float('-inf'), device=q.device, dtype=q.dtype
        )
        acc_sum = torch.zeros(b, h, br, 1, device=q.device, dtype=q.dtype)

        # Iterate over K/V blocks
        for j in range(num_kv_blocks):
            kv_start = j * block_size
            kv_end = min(kv_start + block_size, seq_len)

            # For causal masking: skip K blocks entirely past Q block
            if causal and kv_start > q_end - 1:
                break

            k_block = k[:, :, kv_start:kv_end]  # [B, H, Bc, D]
            v_block = v[:, :, kv_start:kv_end]  # [B, H, Bc, D]

            # Step 1: Compute local attention scores [B, H, Br, Bc]
            scores = (q_block @ k_block.transpose(-2, -1)) * scale

            # Step 2: Apply causal mask if needed
            if causal:
                q_indices = torch.arange(
                    q_start, q_end, device=q.device
                ).unsqueeze(1)
                k_indices = torch.arange(
                    kv_start, kv_end, device=q.device
                ).unsqueeze(0)
                causal_mask = k_indices > q_indices  # [Br, Bc]
                scores = scores.masked_fill(causal_mask, float('-inf'))

            # Step 3: Online softmax -- find block max
            block_max = scores.max(dim=-1, keepdim=True).values
            block_max = block_max.clamp(min=-1e30)

            # Step 4: Compute exp(scores - block_max)
            exp_scores = torch.exp(scores - block_max)  # [B, H, Br, Bc]
            block_sum = exp_scores.sum(dim=-1, keepdim=True)

            # Step 5: Online softmax rescaling
            new_max = torch.maximum(acc_max, block_max)
            old_scale = torch.exp(acc_max - new_max)
            new_scale = torch.exp(block_max - new_max)

            # Step 6: Rescale old output and add new contribution
            # We store the *unnormalized* weighted sum
            acc_output = acc_output * old_scale * acc_sum + new_scale * (
                exp_scores @ v_block
            )

            # Step 7: Update running sum and max
            acc_sum = old_scale * acc_sum + new_scale * block_sum
            acc_max = new_max

            # Normalize by current sum
            acc_output = acc_output / acc_sum

        q_block_outputs.append(acc_output)

    # Concatenate all Q block outputs along the sequence dimension
    return torch.cat(q_block_outputs, dim=2)


class FlashAttention(nn.Module):
    """Multi-Head Attention using the Flash Attention tiling algorithm.

    Drop-in replacement for ``MultiHeadAttention`` that uses the tiled
    online softmax algorithm from Dao et al. (2022) instead of
    materializing the full attention matrix.

    The output is mathematically identical to standard MHA. The
    educational value is in demonstrating *how* tiling + online softmax
    avoids the O(N^2) memory bottleneck.

    Args:
        model_config: Model configuration (uses embed_dim, num_heads,
            context_length, qkv_bias).
        block_size: Tile size for the Flash Attention algorithm.
    """

    causal_mask: torch.Tensor

    def __init__(
        self,
        model_config: ModelConfig,
        block_size: int = 64,
    ) -> None:
        """Initialize Flash Attention layer.

        Args:
            model_config: Model configuration.
            block_size: Block/tile size for the tiling algorithm.
        """
        super().__init__()
        assert model_config.embed_dim % model_config.num_heads == 0, (
            'embed_dim must be divisible by num_heads'
        )
        self.embed_dim = model_config.embed_dim
        self.num_heads = model_config.num_heads
        self.head_dim = model_config.embed_dim // model_config.num_heads
        self.block_size = block_size

        self.qkv_proj = nn.Linear(
            model_config.embed_dim,
            3 * model_config.embed_dim,
            bias=model_config.qkv_bias,
        )
        self.out_proj = nn.Linear(
            model_config.embed_dim, model_config.embed_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Flash Attention.

        Args:
            x: Input tensor ``[batch, seq_len, embed_dim]``.

        Returns:
            Output tensor ``[batch, seq_len, embed_dim]``.
        """
        b, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Reshape to [B, H, L, D]
        queries = queries.view(
            b, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(b, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        values = values.view(
            b, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Apply flash attention
        context = flash_attention(
            queries,
            keys,
            values,
            block_size=self.block_size,
            causal=True,
        )

        # Reshape back to [B, L, embed_dim]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(b, seq_len, self.embed_dim)
        )

        return self.out_proj(context)
