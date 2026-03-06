r"""LLaMA (Large Language Model Meta AI) architecture.

Implements the decoder-only transformer from:
    Touvron et al., 2023 -- "LLaMA: Open and Efficient Foundation
    Language Models"

Key differences from GPT-2:
- **RMSNorm** instead of LayerNorm (simpler, ~same quality)
- **RoPE** instead of learned positional embeddings (relative pos)
- **SwiGLU** instead of GELU FFN (gated activation)
- **GQA** instead of standard MHA (efficient KV cache)
- **No bias** in linear layers
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from lmt.layers.ffn import SwiGLU
from lmt.layers.normalization import RMSNorm
from lmt.layers.positional import RoPE
from lmt.models.config import ModelConfig


class LlamaAttention(nn.Module):
    """GQA with integrated RoPE for the LLaMA architecture.

    This combines Grouped Query Attention with Rotary Positional
    Encoding applied to Q and K before the dot product.
    """

    causal_mask: Tensor

    def __init__(self, config: ModelConfig, rope: RoPE) -> None:
        """Initialize LlamaAttention.

        Args:
            config: Model configuration.
            rope: Shared RoPE instance for this model.
        """
        super().__init__()
        num_kv_heads = config.num_kv_heads or config.num_heads

        assert config.embed_dim % config.num_heads == 0
        assert config.num_heads % num_kv_heads == 0

        self.num_heads = config.num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_groups = config.num_heads // num_kv_heads
        self.rope = rope

        self.q_proj = nn.Linear(
            config.embed_dim,
            config.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.embed_dim,
            num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.embed_dim,
            num_kv_heads * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            config.embed_dim, config.embed_dim, bias=False
        )

        self.register_buffer(
            'causal_mask',
            torch.full(
                (config.context_length, config.context_length),
                float('-inf'),
            ).triu(diagonal=1),
        )
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply GQA with RoPE.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output with same shape.
        """
        b, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [b, seq, heads, head_dim]
        q = q.view(b, seq_len, self.num_heads, self.head_dim)
        k = k.view(b, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(b, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K per head
        # Reshape to [b*heads, seq, head_dim] for RoPE, then back
        q = q.permute(0, 2, 1, 3).reshape(
            b * self.num_heads, seq_len, self.head_dim
        )
        k = k.permute(0, 2, 1, 3).reshape(
            b * self.num_kv_heads, seq_len, self.head_dim
        )
        q, k = self.rope.apply_rotary_emb(q, k)

        # Back to [b, heads, seq, head_dim]
        q = q.view(b, self.num_heads, seq_len, self.head_dim)
        k = k.view(b, self.num_kv_heads, seq_len, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # [b, kv_heads, seq, head_dim]

        # Expand KV heads
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:seq_len, :seq_len]
        attn = attn + mask
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)

        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, seq_len, -1)
        return self.out_proj(out)


class LlamaBlock(nn.Module):
    """A single LLaMA transformer block.

    Pre-norm architecture with RMSNorm, GQA+RoPE, and SwiGLU.
    """

    def __init__(self, config: ModelConfig, rope: RoPE) -> None:
        """Initialize LlamaBlock.

        Args:
            config: Model configuration.
            rope: Shared RoPE instance.
        """
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim)
        self.attn = LlamaAttention(config, rope)
        self.ffn_norm = RMSNorm(config.embed_dim)
        self.ffn = SwiGLU(d_model=config.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply LLaMA block.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output with same shape.
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLaMA(nn.Module):
    """LLaMA decoder-only transformer model.

    Composes token embedding, RoPE, stacked LlamaBlocks,
    final RMSNorm, and output projection.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize LLaMA model.

        Args:
            config: Model configuration with embed_dim, num_heads,
                num_kv_heads, context_length, vocab_size, num_layers.
        """
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(
            d_model=head_dim,
            max_seq_len=config.context_length,
        )

        self.blocks = nn.ModuleList(
            [LlamaBlock(config, rope) for _ in range(config.num_layers)]
        )

        self.final_norm = RMSNorm(config.embed_dim)
        self.out_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with scaled residual init."""
        num_layers = len(self.blocks)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scaled init for residual projections
        std = 0.02 / math.sqrt(2 * num_layers)
        for module in self.blocks:
            block: LlamaBlock = module  # type: ignore[assignment]
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=std)

    def forward(self, in_idx: Tensor) -> Tensor:
        """Forward pass of LLaMA model.

        Args:
            in_idx: Token indices ``[batch, seq_len]``.

        Returns:
            Logits of shape ``[batch, seq_len, vocab_size]``.
        """
        x = self.tok_embed(in_idx)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.out_head(x)
