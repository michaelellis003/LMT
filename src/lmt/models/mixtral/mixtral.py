r"""Mixtral architecture (Mixture of Experts).

Implements the sparse MoE transformer from:
    Jiang et al., 2024 -- "Mixtral of Experts"

Mixtral is LLaMA with two key modifications:
- **MoE FFN**: each layer's dense SwiGLU is replaced by a
  Mixture of Experts layer with top-k routing
- **Sliding Window Attention**: each token attends to the
  previous ``w`` tokens only, with effective receptive field
  growing linearly with depth

The aux_loss from all MoE layers is summed and stored as
``self.aux_loss`` for the training loop.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from lmt.layers.ffn.moe import MoEFeedForward
from lmt.layers.normalization import RMSNorm
from lmt.layers.positional import RoPE
from lmt.models.config import ModelConfig


class MixtralAttention(nn.Module):
    """Sliding window GQA with RoPE for Mixtral."""

    causal_mask: Tensor

    def __init__(
        self, config: ModelConfig, rope: RoPE
    ) -> None:
        """Initialize MixtralAttention.

        Args:
            config: Model configuration.
            rope: Shared RoPE instance.
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

        # Build sliding window causal mask
        ctx = config.context_length
        w = config.window_size or ctx
        mask = torch.full((ctx, ctx), float('-inf'))
        for i in range(ctx):
            start = max(0, i - w + 1)
            mask[i, start : i + 1] = 0.0
        self.register_buffer('causal_mask', mask)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply sliding window GQA with RoPE.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output with same shape.
        """
        b, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, seq_len, self.num_heads, self.head_dim)
        k = k.view(b, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(b, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
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

        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:seq_len, :seq_len]
        attn = attn + mask
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)

        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, seq_len, -1)
        return self.out_proj(out)


class MixtralBlock(nn.Module):
    """A single Mixtral block: RMSNorm + SWA+RoPE + MoE."""

    def __init__(
        self,
        config: ModelConfig,
        rope: RoPE,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        """Initialize MixtralBlock.

        Args:
            config: Model configuration.
            rope: Shared RoPE instance.
            num_experts: Number of MoE experts.
            top_k: Experts per token.
        """
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim)
        self.attn = MixtralAttention(config, rope)
        self.ffn_norm = RMSNorm(config.embed_dim)
        self.ffn = MoEFeedForward(
            d_model=config.embed_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply Mixtral block.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output with same shape.
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Mixtral(nn.Module):
    """Mixtral sparse MoE transformer model.

    LLaMA architecture with MoE FFN and sliding window attention.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        """Initialize Mixtral model.

        Args:
            config: Model configuration.
            num_experts: Experts per MoE layer.
            top_k: Experts activated per token.
        """
        super().__init__()
        self.tok_embed = nn.Embedding(
            config.vocab_size, config.embed_dim
        )

        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(
            d_model=head_dim,
            max_seq_len=config.context_length,
        )

        self.blocks = nn.ModuleList(
            [
                MixtralBlock(config, rope, num_experts, top_k)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(config.embed_dim)
        self.out_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False
        )
        self.aux_loss = torch.tensor(0.0)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with scaled residual init."""
        num_layers = len(self.blocks)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        std = 0.02 / math.sqrt(2 * num_layers)
        for module in self.blocks:
            block: MixtralBlock = module  # type: ignore[assignment]
            nn.init.normal_(
                block.attn.out_proj.weight, mean=0.0, std=std
            )

    def forward(self, in_idx: Tensor) -> Tensor:
        """Forward pass of Mixtral.

        Args:
            in_idx: Token indices ``[batch, seq_len]``.

        Returns:
            Logits ``[batch, seq_len, vocab_size]``.
        """
        x = self.tok_embed(in_idx)
        total_aux = torch.tensor(0.0, device=x.device)
        for module in self.blocks:
            blk: MixtralBlock = module  # type: ignore[assignment]
            x = blk(x)
            total_aux = total_aux + blk.ffn.aux_loss
        self.aux_loss = total_aux
        x = self.final_norm(x)
        return self.out_head(x)
