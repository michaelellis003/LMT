r"""Multi-Head Latent Attention (MLA) with Decoupled RoPE.

Implements the attention mechanism from:
    DeepSeek-AI, 2024 -- "DeepSeek-V2: A Strong, Economical, and
    Efficient Mixture-of-Experts Language Model"

MLA compresses key-value representations into a low-rank latent
space before attention, dramatically reducing KV cache size.

The key innovation is **decoupled RoPE**: positional information
is handled by separate small vectors (q_R, k_R) while content
uses the compressed latent path. This solves the incompatibility
between low-rank KV compression and rotary position encoding.

.. math::

    \text{Content: } c_t^{KV} = x_t W^{DKV}, \quad
    k_C = c^{KV} W^{UK}, \quad v = c^{KV} W^{UV}

    \text{Position: } k_R = \text{RoPE}(x_t W^{KR})

    \text{Query: } c^Q = x_t W^{DQ}, \quad
    q_C = c^Q W^{UQ}, \quad q_R = \text{RoPE}(c^Q W^{QR})

    Q = [q_C; q_R], \quad K = [k_C; k_R]

At inference, only :math:`c_t^{KV}` and :math:`k_R` are cached,
giving massive memory savings over standard MHA.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from lmt.layers.attention.kv_cache import KVCache
from lmt.layers.positional import RoPE
from lmt.models.config import ModelConfig


class MultiHeadLatentAttention(nn.Module):
    r"""Multi-Head Latent Attention with KV compression and decoupled RoPE.

    Args:
        model_config: Model configuration.
        kv_compress_dim: Latent dimension for KV compression.
        q_compress_dim: Latent dimension for Q compression.
        rope_dim: Dimension for decoupled RoPE vectors.
            If 0, RoPE is disabled (pure content attention).
    """

    causal_mask: Tensor

    def __init__(
        self,
        model_config: ModelConfig,
        kv_compress_dim: int,
        q_compress_dim: int,
        rope_dim: int = 0,
    ) -> None:
        """Initialize MLA.

        Args:
            model_config: Model configuration.
            kv_compress_dim: KV latent dimension.
            q_compress_dim: Q latent dimension.
            rope_dim: Dimension for decoupled RoPE (0 = disabled).
        """
        super().__init__()
        embed_dim = model_config.embed_dim
        num_heads = model_config.num_heads
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_compress_dim = kv_compress_dim
        self.rope_dim = rope_dim

        # Effective head dim for Q@K includes RoPE dimensions
        qk_dim = self.head_dim + rope_dim
        self.scale = 1.0 / math.sqrt(qk_dim)

        # Q content path: x -> compress -> decompress
        self.w_dq = nn.Linear(embed_dim, q_compress_dim, bias=False)
        self.w_uq = nn.Linear(
            q_compress_dim, num_heads * self.head_dim, bias=False
        )

        # KV path: x -> compress -> decompress to K_content and V
        self.w_dkv = nn.Linear(embed_dim, kv_compress_dim, bias=False)
        self.w_uk = nn.Linear(
            kv_compress_dim, num_heads * self.head_dim, bias=False
        )
        self.w_uv = nn.Linear(
            kv_compress_dim, num_heads * self.head_dim, bias=False
        )

        # Decoupled RoPE projections (only if rope_dim > 0)
        if rope_dim > 0:
            # q_R derived from compressed query latent
            self.w_qr = nn.Linear(
                q_compress_dim, num_heads * rope_dim, bias=False
            )
            # k_R derived from input directly (NOT from KV latent)
            self.w_kr = nn.Linear(embed_dim, num_heads * rope_dim, bias=False)
            self.rope = RoPE(
                d_model=rope_dim,
                max_seq_len=model_config.context_length,
            )

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.register_buffer(
            'causal_mask',
            torch.full(
                (model_config.context_length, model_config.context_length),
                float('-inf'),
            ).triu(diagonal=1),
        )

        # Optional KV cache for autoregressive generation.
        self.kv_cache: KVCache | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head latent attention with decoupled RoPE.

        When ``kv_cache`` is set, new K/V are appended to the cache
        and the full cached K/V are used for attention.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output with same shape.
        """
        b, seq_len, _ = x.shape

        # Track position offset for RoPE when using cache
        pos_offset = self.kv_cache.seq_len if self.kv_cache is not None else 0

        # Compress query
        c_q = self.w_dq(x)  # [b, seq, q_compress_dim]

        # Content Q: decompress from query latent
        q_c = self.w_uq(c_q)  # [b, seq, num_heads * head_dim]
        q_c = q_c.view(b, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Compress KV
        c_kv = self.w_dkv(x)  # [b, seq, kv_compress_dim]

        # Content K and V: decompress from KV latent
        k_c = self.w_uk(c_kv)  # [b, seq, num_heads * head_dim]
        v = self.w_uv(c_kv)  # [b, seq, num_heads * head_dim]

        k_c = k_c.view(b, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope_dim > 0:
            # Positional Q: from query latent
            q_r = self.w_qr(c_q)  # [b, seq, num_heads * rope_dim]
            q_r = q_r.view(
                b, seq_len, self.num_heads, self.rope_dim
            ).transpose(1, 2)

            # Positional K: from input directly (asymmetric!)
            k_r = self.w_kr(x)  # [b, seq, num_heads * rope_dim]
            k_r = k_r.view(
                b, seq_len, self.num_heads, self.rope_dim
            ).transpose(1, 2)

            # Apply RoPE with position offset for cached generation
            q_r = q_r.reshape(b * self.num_heads, seq_len, self.rope_dim)
            k_r = k_r.reshape(b * self.num_heads, seq_len, self.rope_dim)
            q_r, k_r = self.rope.apply_rotary_emb(q_r, k_r, offset=pos_offset)
            q_r = q_r.view(b, self.num_heads, seq_len, self.rope_dim)
            k_r = k_r.view(b, self.num_heads, seq_len, self.rope_dim)

            # Concatenate content and positional: [head_dim + rope_dim]
            q = torch.cat([q_c, q_r], dim=-1)
            k = torch.cat([k_c, k_r], dim=-1)
        else:
            q = q_c
            k = k_c

        # Update KV cache if active
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(k, v)

        # Scaled dot-product attention with causal mask
        kv_len = k.shape[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        q_start = kv_len - seq_len
        mask = self.causal_mask[q_start:kv_len, :kv_len]
        attn = attn + mask
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)

        # Attention applied to V (content only, no positional component)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)
        return self.out_proj(out)
