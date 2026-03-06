r"""Multi-Head Latent Attention (MLA).

Implements the attention mechanism from:
    DeepSeek-AI, 2024 -- "DeepSeek-V2: A Strong, Economical, and
    Efficient Mixture-of-Experts Language Model"

MLA compresses key-value representations into a low-rank latent
space before attention, dramatically reducing KV cache size:

.. math::

    c_t^{KV} = x_t W^{DKV} \quad \text{(compress to latent)}

    [k_t, v_t] = c_t^{KV} W^{UKV} \quad \text{(decompress for attn)}

Query is similarly compressed:

.. math::

    c_t^Q = x_t W^{DQ}, \quad q_t = c_t^Q W^{UQ}

At inference time, only :math:`c_t^{KV}` is cached instead of
the full K,V tensors. If ``kv_compress_dim << n_heads * head_dim``,
this gives massive memory savings.

The key insight: K and V share the same compressed representation,
so the bottleneck forces the model to learn a joint low-rank
factorization of the key-value space.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from lmt.models.config import ModelConfig


class MultiHeadLatentAttention(nn.Module):
    r"""Multi-Head Latent Attention with KV compression.

    Args:
        model_config: Model configuration.
        kv_compress_dim: Latent dimension for KV compression.
        q_compress_dim: Latent dimension for Q compression.
    """

    causal_mask: Tensor

    def __init__(
        self,
        model_config: ModelConfig,
        kv_compress_dim: int,
        q_compress_dim: int,
    ) -> None:
        """Initialize MLA.

        Args:
            model_config: Model configuration.
            kv_compress_dim: KV latent dimension.
            q_compress_dim: Q latent dimension.
        """
        super().__init__()
        embed_dim = model_config.embed_dim
        num_heads = model_config.num_heads
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_compress_dim = kv_compress_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q path: x -> compress -> decompress to full Q
        self.w_dq = nn.Linear(embed_dim, q_compress_dim, bias=False)
        self.w_uq = nn.Linear(
            q_compress_dim, num_heads * self.head_dim, bias=False
        )

        # KV path: x -> compress -> decompress to K and V
        self.w_dkv = nn.Linear(embed_dim, kv_compress_dim, bias=False)
        self.w_ukv = nn.Linear(
            kv_compress_dim, 2 * num_heads * self.head_dim, bias=False
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.register_buffer(
            'causal_mask',
            torch.full(
                (model_config.context_length, model_config.context_length),
                float('-inf'),
            ).triu(diagonal=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head latent attention.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output with same shape.
        """
        b, seq_len, _ = x.shape

        # Compress then decompress Q
        q = self.w_uq(self.w_dq(x))

        # Compress then decompress KV
        c_kv = self.w_dkv(x)  # [b, seq, kv_compress_dim] -- the latent
        kv = self.w_ukv(c_kv)  # [b, seq, 2 * num_heads * head_dim]
        k, v = kv.chunk(2, dim=-1)

        # Reshape to multi-head: [b, heads, seq, head_dim]
        q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:seq_len, :seq_len]
        attn = attn + mask
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)
        return self.out_proj(out)
